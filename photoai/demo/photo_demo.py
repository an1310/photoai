import time

import gradio as gr
import numpy as np
import torch

from base_demo import ImageAIDemoBase, create_base_parser, get_config_path, setup_model_optimizations
from photoai.util import HWC3, upscale_image


class UnifiedPhotoAIDemo(ImageAIDemoBase):
    """Unified PhotoAI demo with support for all variants and tiled processing"""

    def __init__(self, args):
        # Enable tiling for tiled variant by default
        if args.model_variant == 'tiled':
            args.enable_tiling = True
            args.use_tile_vae = True
            self.tile_size = 128  # Larger tiles for tiled variant
            self.tile_stride = 64
        elif args.model_variant == 'lightning':
            # Lightning optimizations
            self.tile_size = 96  # Smaller tiles for speed
            self.tile_stride = 48
        else:
            # Base model settings
            self.tile_size = getattr(args, 'tile_size', 128)
            self.tile_stride = getattr(args, 'tile_stride', 64)

        super().__init__(args)
        self.interface = None
        self.current_model_cache = {}

    def setup_imageai_model(self):
        """Setup PhotoAI model using selected variant"""
        print(f"Loading PhotoAI model variant: {self.args.model_variant}")

        # Get config path based on variant
        config_path = get_config_path(self.args.model_variant)
        print(f"Using config: {config_path}")

        try:
            # Create model from config
            self.model = self.load_config_and_create_model(
                config_path=config_path,
                device=self.imageai_device
            )

            # Apply optimizations
            self.model = setup_model_optimizations(self.model, self.args)

            # Enable tiled VAE by default for tiled variant
            if self.args.model_variant == 'tiled' and not self.args.use_tile_vae:
                if hasattr(self.model, 'init_tile_vae'):
                    self.model.init_tile_vae(encoder_tile_size=512, decoder_tile_size=64)
                    print("✅ Auto-enabled tiled VAE for tiled variant")

            print("✅ PhotoAI model loaded successfully!")

            # Store model info
            model_info = self.get_model_info()
            print(f"Model type: {model_info['model_type']}")
            print(f"Scheduler: {model_info['scheduler']}")
            print(f"VAE: {model_info['vae_type']}")
            if self.enable_tiling:
                print(f"Tiling: {self.tile_size} (latent) / ~{self.tile_size * 8}px (image)")

        except Exception as e:
            print(f"❌ Failed to load PhotoAI model: {e}")
            raise

    def stage2_process(self, input_image, prompt, a_prompt, n_prompt, num_samples,
                       upscale_factor, num_steps, guidance_scale, control_scale, seed,
                       color_fix_type, use_linear_cfg=False, guidance_scale_min=4.0,
                       enable_tiling=None, tile_size=None, tile_stride=None,
                       restoration_scale=4.0, s_churn=0.0, s_noise=1.003, photoai_checkpoint="Q",
                       enable_multipass=False, num_passes=2):
        """Stage 2 processing with optional tiling and multi-pass - compatible with PhotoAI"""

        if input_image is None:
            return [], "No input image provided"

        # Update tiling settings if provided
        if enable_tiling is not None:
            self.enable_tiling = enable_tiling
        if tile_size is not None:
            self.tile_size = tile_size
        if tile_stride is not None:
            self.tile_stride = tile_stride

        # Handle checkpoint switching
        if hasattr(self.model, 'current_model') and self.model.current_model != f'v0-{photoai_checkpoint}':
            try:
                # Load the requested checkpoint
                from photoai.util import load_QF_ckpt
                ckpt_Q, ckpt_F = load_QF_ckpt('options/PhotoAI_v0.yaml')

                if photoai_checkpoint == 'Q':
                    self.model.load_state_dict(ckpt_Q, strict=False)
                    self.model.current_model = 'v0-Q'
                    print(f"✅ Switched to Quality checkpoint (Q)")
                elif photoai_checkpoint == 'F':
                    self.model.load_state_dict(ckpt_F, strict=False)
                    self.model.current_model = 'v0-F'
                    print(f"✅ Switched to Fidelity checkpoint (F)")
            except Exception as e:
                print(f"⚠️ Failed to switch checkpoint: {e}")

        # Create event for logging
        event_id = str(time.time_ns())
        event_dict = {
            'event_id': event_id,
            'localtime': time.ctime(),
            'model_variant': self.args.model_variant,
            'prompt': prompt,
            'a_prompt': a_prompt,
            'n_prompt': n_prompt,
            'num_samples': num_samples,
            'upscale_factor': upscale_factor,
            'num_steps': num_steps,
            'guidance_scale': guidance_scale,
            'control_scale': control_scale,
            'seed': seed,
            'color_fix_type': color_fix_type,
            'use_linear_cfg': use_linear_cfg,
            'enable_tiling': self.enable_tiling,
            'tile_size': self.tile_size,
            'tile_stride': self.tile_stride,
            'restoration_scale': restoration_scale,
            's_churn': s_churn,
            's_noise': s_noise,
            'photoai_checkpoint': photoai_checkpoint,
            'enable_multipass': enable_multipass,
            'num_passes': num_passes
        }

        try:
            torch.cuda.set_device(self.imageai_device)

            # Prepare input image
            input_image = HWC3(input_image)
            if upscale_factor > 1:
                input_image = upscale_image(input_image, upscale_factor, unit_resolution=32, min_size=1024)

            # Prepare image tensor
            LQ = np.array(input_image) / 255.0
            LQ = LQ * 2.0 - 1.0  # Normalize to [-1, 1]
            LQ = torch.tensor(LQ, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.imageai_device)[:, :3, :,
                 :]

            # Prepare captions/prompts
            captions = []
            for _ in range(num_samples):
                if prompt.strip():
                    full_prompt = prompt.strip()
                    if a_prompt.strip():
                        full_prompt = f"{full_prompt}, {a_prompt.strip()}"
                else:
                    full_prompt = a_prompt.strip() if a_prompt.strip() else ""
                captions.append(full_prompt)

            # Set model dtypes if available
            if hasattr(self.model, 'ae_dtype'):
                self.model.ae_dtype = torch.float32  # or convert_dtype based on your needs
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'dtype'):
                self.model.model.dtype = torch.float32

            # Multi-pass processing
            if enable_multipass and num_passes > 1:
                print(f"🔄 Multi-pass processing: {num_passes} passes")

                # Progressive enhancement strategy
                current_image = LQ

                for pass_num in range(num_passes):
                    print(f"  Pass {pass_num + 1}/{num_passes}")

                    # Adjust parameters for each pass
                    pass_restoration = restoration_scale * (0.4 + 0.6 * (pass_num + 1) / num_passes)

                    # Process this pass
                    with torch.no_grad():
                        if self.enable_tiling and hasattr(self.model, 'batchify_sample_tiled'):
                            samples = self.model.batchify_sample_tiled(
                                current_image, captions,
                                num_steps=num_steps,
                                restoration_scale=pass_restoration,
                                s_churn=s_churn,
                                s_noise=s_noise,
                                cfg_scale=guidance_scale,
                                control_scale=control_scale,
                                seed=seed if pass_num == 0 else -1,  # Only use seed on first pass
                                num_samples=num_samples,
                                p_p=a_prompt,
                                n_p=n_prompt,
                                color_fix_type=color_fix_type,
                                use_linear_CFG=use_linear_cfg,
                                use_linear_control_scale=False,
                                cfg_scale_start=guidance_scale,
                                control_scale_start=control_scale,
                                tile_size=self.tile_size,
                                tile_stride=self.tile_stride
                            )
                        else:
                            samples = self.model.batchify_sample(
                                current_image, captions,
                                num_steps=num_steps,
                                restoration_scale=pass_restoration,
                                s_churn=s_churn,
                                s_noise=s_noise,
                                cfg_scale=guidance_scale,
                                control_scale=control_scale,
                                seed=seed if pass_num == 0 else -1,
                                num_samples=num_samples,
                                p_p=a_prompt,
                                n_p=n_prompt,
                                color_fix_type=color_fix_type,
                                use_linear_CFG=use_linear_cfg,
                                use_linear_control_scale=False,
                                cfg_scale_start=guidance_scale,
                                control_scale_start=control_scale
                            )

                    # Prepare for next pass (convert back to tensor if needed)
                    if pass_num < num_passes - 1:
                        if isinstance(samples, torch.Tensor):
                            current_image = samples
                        else:
                            # Convert numpy back to tensor if needed
                            if isinstance(samples, list):
                                samples_array = np.array(samples[0]) if samples else current_image.cpu().numpy()
                            else:
                                samples_array = samples
                            samples_normalized = (samples_array.astype(
                                np.float32) / 255.0) * 2.0 - 1.0  # Back to [-1,1]
                            current_image = torch.tensor(samples_normalized, dtype=torch.float32).permute(2, 0,
                                                                                                          1).unsqueeze(
                                0).to(self.imageai_device)

                # Use final pass result
                final_samples = samples

            else:
                # Single pass processing (original logic)
                with torch.no_grad():
                    if self.enable_tiling and hasattr(self.model, 'batchify_sample_tiled'):
                        final_samples = self.model.batchify_sample_tiled(
                            LQ, captions,
                            num_steps=num_steps,
                            restoration_scale=restoration_scale,
                            s_churn=s_churn,
                            s_noise=s_noise,
                            cfg_scale=guidance_scale,
                            control_scale=control_scale,
                            seed=seed,
                            num_samples=num_samples,
                            p_p=a_prompt,
                            n_p=n_prompt,
                            color_fix_type=color_fix_type,
                            use_linear_CFG=use_linear_cfg,
                            use_linear_control_scale=False,
                            cfg_scale_start=guidance_scale,
                            control_scale_start=control_scale,
                            tile_size=self.tile_size,
                            tile_stride=self.tile_stride
                        )
                    else:
                        final_samples = self.model.batchify_sample(
                            LQ, captions,
                            num_steps=num_steps,
                            restoration_scale=restoration_scale,
                            s_churn=s_churn,
                            s_noise=s_noise,
                            cfg_scale=guidance_scale,
                            control_scale=control_scale,
                            seed=seed,
                            num_samples=num_samples,
                            p_p=a_prompt,
                            n_p=n_prompt,
                            color_fix_type=color_fix_type,
                            use_linear_CFG=use_linear_cfg,
                            use_linear_control_scale=False,
                            cfg_scale_start=guidance_scale,
                            control_scale_start=control_scale
                        )

            # Convert results to list of numpy arrays
            if isinstance(final_samples, torch.Tensor):
                samples_np = final_samples.detach().cpu().float().numpy()
                samples_np = (samples_np + 1.0) / 2.0  # Denormalize from [-1,1] to [0,1]
                samples_np = samples_np.transpose(0, 2, 3, 1)  # BCHW to BHWC
                samples_np = (samples_np * 255.0).round().clip(0, 255).astype(np.uint8)
                results = [samples_np[i] for i in range(samples_np.shape[0])]
            else:
                results = final_samples  # Assume already in correct format

            # Log event
            self.log_event(event_dict)

            return results, event_id

        except Exception as e:
            error_msg = f"Error in stage 2 processing: {str(e)}"
            print(error_msg)
            return [], error_msg

    def setup_gradio_interface(self):
        """Setup unified Gradio interface with all features"""

        def create_interface():
            title = f"PhotoAI Unified Demo - {self.args.model_variant.capitalize()} Variant"
            with gr.Blocks(title=title, theme=gr.themes.Soft()) as interface:
                gr.Markdown(f"# {title}")
                gr.Markdown(
                    f"**Config:** {get_config_path(self.args.model_variant)} | **Tiling Available:** {'✅' if self.enable_tiling else '❌'}")

                with gr.Tab("🚀 Complete Pipeline"):
                    with gr.Row():
                        with gr.Column():
                            # Get common components
                            components = self.create_common_interface_components()

                            # Additional prompts
                            with gr.Accordion("Advanced Prompts", open=False):
                                a_prompt = gr.Textbox(
                                    label="Additional Prompt",
                                    value="detailed, high quality, sharp focus, professional photography",
                                    lines=2
                                )
                                n_prompt = gr.Textbox(
                                    label="Negative Prompt",
                                    value="blurry, low quality, distorted, watermark, signature",
                                    lines=2
                                )

                            # Advanced settings - Combined
                            with gr.Accordion("Advanced Settings", open=False):
                                with gr.Row():
                                    num_samples = gr.Slider(1, 4, value=1, step=1, label="Number of Samples")
                                    use_linear_cfg = gr.Checkbox(value=False, label="Linear CFG")

                                guidance_scale_min = gr.Slider(
                                    1.0, 10.0, value=4.0, label="Min CFG Scale (for Linear CFG)"
                                )

                                gr.Markdown("---")
                                gr.Markdown("**Multi-Pass Enhancement**")

                                with gr.Row():
                                    enable_multipass = gr.Checkbox(
                                        value=False,
                                        label="Enable Multi-Pass Processing",
                                        info="Process image multiple times for dramatic enhancement"
                                    )
                                    num_passes = gr.Slider(
                                        1, 4, value=2, step=1,
                                        label="Number of Passes",
                                        info="More passes = more dramatic results"
                                    )

                                gr.Markdown("---")
                                gr.Markdown("**PhotoAI Enhancement Parameters**")

                                with gr.Row():
                                    restoration_scale = gr.Slider(
                                        1.0, 8.0, value=4.0, step=0.1,
                                        label="Restoration Scale",
                                        info="Higher values = more aggressive enhancement"
                                    )
                                    photoai_checkpoint = gr.Radio(
                                        choices=["Q", "F"], value="Q",
                                        label="PhotoAI Checkpoint",
                                        info="Q=Quality focused, F=Fidelity focused"
                                    )

                                with gr.Row():
                                    s_churn = gr.Slider(
                                        0.0, 2.0, value=0.0, step=0.1,
                                        label="S-Churn",
                                        info="Stochasticity for detail generation"
                                    )
                                    s_noise = gr.Slider(
                                        0.5, 2.0, value=1.003, step=0.001,
                                        label="S-Noise",
                                        info="Noise injection for texture detail"
                                    )

                                with gr.Accordion("💡 Enhancement Tips", open=False):
                                    gr.Markdown("""
                                    **Multi-Pass Processing:**
                                    - **2 Passes:** Conservative progressive enhancement
                                    - **3+ Passes:** Dramatic transformation for heavily damaged images
                                    - Each pass builds on the previous result
                                    - Safer than single extreme settings

                                    **For Dramatic Results:**
                                    - **Restoration Scale 6-8:** Very aggressive enhancement
                                    - **S-Churn 0.5-1.0:** Adds detail variation  
                                    - **S-Noise 1.5-2.0:** More texture detail
                                    - **Q Checkpoint:** Better for overall quality
                                    - **F Checkpoint:** Better for preserving original details

                                    **Conservative/Safe:**
                                    - **Restoration Scale 2-4:** Moderate enhancement
                                    - **S-Churn 0.0:** No randomness
                                    - **S-Noise 1.0:** Minimal noise injection
                                    """)

                            # Process button
                            process_btn = gr.Button("🎨 Generate", variant="primary", size="lg")

                        with gr.Column():
                            # Outputs
                            result_gallery = gr.Gallery(
                                label="Generated Images",
                                columns=2,
                                rows=2,
                                height=500
                            )
                            event_id_output = gr.Textbox(label="Event ID", interactive=False)

                    # Button click handler
                    process_btn.click(
                        self.stage2_process,
                        inputs=[
                            components['input_image'], components['prompt'], a_prompt, n_prompt,
                            num_samples, components['upscale_factor'], components['num_steps'],
                            components['guidance_scale'], components['control_scale'], components['seed'],
                            components['color_fix_type'], use_linear_cfg, guidance_scale_min,
                            components['enable_tiling'], components['tile_size'], components['tile_stride'],
                            restoration_scale, s_churn, s_noise, photoai_checkpoint,
                            enable_multipass, num_passes
                        ],
                        outputs=[result_gallery, event_id_output]
                    )

                with gr.Tab("🔧 Stage 1 - Enhancement"):
                    with gr.Row():
                        with gr.Column():
                            stage1_input = gr.Image(label="Input Image", type="numpy")
                            gamma_correction = gr.Slider(0.5, 2.0, value=1.0, label="Gamma Correction")
                            stage1_btn = gr.Button("Enhance Image", variant="primary")

                        with gr.Column():
                            stage1_output = gr.Image(label="Enhanced Image", type="numpy")

                    stage1_btn.click(
                        self.stage1_process,
                        inputs=[stage1_input, gamma_correction],
                        outputs=[stage1_output]
                    )

                with gr.Tab("🧠 LLaVA Analysis") if self.use_llava else gr.Tab("🧠 LLaVA Analysis", visible=False):
                    with gr.Row():
                        with gr.Column():
                            llava_input = gr.Image(label="Image to Analyze", type="numpy")

                            llava_prompt = gr.Textbox(
                                label="Analysis Prompt",
                                value="Describe this image in detail for image enhancement guidance.",
                                lines=3
                            )

                            with gr.Row():
                                llava_temp = gr.Slider(0.1, 1.0, value=0.7, label="Temperature")
                                llava_top_p = gr.Slider(0.1, 1.0, value=0.9, label="Top P")

                            llava_btn = gr.Button("Analyze Image", variant="secondary")

                        with gr.Column():
                            llava_output = gr.Textbox(
                                label="Analysis Result",
                                lines=10,
                                max_lines=20
                            )

                    llava_btn.click(
                        self.llava_caption_process,
                        inputs=[llava_input, llava_temp, llava_top_p, llava_prompt],
                        outputs=[llava_output]
                    )

                with gr.Tab("🔍 Tile Visualization"):
                    gr.Markdown("Visualize how images will be processed with tiled approach")

                    with gr.Row():
                        with gr.Column():
                            viz_image = gr.Image(label="Image for Tile Analysis", type="numpy")
                            show_grid = gr.Checkbox(value=True, label="Show Tile Grid")

                            with gr.Accordion("Custom Tile Settings", open=False):
                                viz_tile_size = gr.Slider(
                                    32, 256, value=self.tile_size, step=16,
                                    label="Custom Tile Size (latent)"
                                )
                                viz_tile_stride = gr.Slider(
                                    16, 128, value=self.tile_stride, step=8,
                                    label="Custom Tile Stride (latent)"
                                )
                                gr.Markdown("*Multiply by ~8 for image space dimensions*")

                            viz_btn = gr.Button("Analyze Tiles", variant="secondary")

                        with gr.Column():
                            grid_output = gr.Image(label="Tile Grid Visualization")
                            tile_count_output = gr.Number(label="Estimated Tile Count", precision=0)

                            with gr.Accordion("Tile Information", open=True):
                                tile_info_md = gr.Markdown("Upload an image to see tile analysis")

                    def update_tile_info(image, show_grid, tile_size, tile_stride):
                        if image is None:
                            return image, 0, "Upload an image to see tile analysis"

                        viz_result, count = self.analyze_tile_coverage(image, show_grid, tile_size, tile_stride)

                        # Calculate some statistics
                        h, w = image.shape[:2]
                        image_tile_size = tile_size * 8
                        image_tile_stride = tile_stride * 8
                        overlap = tile_size - tile_stride

                        info = f"""
                        **Image Dimensions:** {w} x {h} pixels  
                        **Tile Count:** {count}  
                        **Tile Size:** {tile_size} (latent) / {image_tile_size} (image)  
                        **Tile Stride:** {tile_stride} (latent) / {image_tile_stride} (image)  
                        **Overlap:** {overlap} (latent) / {overlap * 8} (image)  
                        **Memory Efficiency:** {'High' if count > 4 else 'Standard'}  

                        💡 **Tips:**
                        - More tiles = lower VRAM usage but slower processing
                        - Larger overlap = better blending but more computation
                        - For 4K+ images, consider tile_size=96, stride=48
                        """

                        return viz_result, count, info

                    viz_btn.click(
                        update_tile_info,
                        inputs=[viz_image, show_grid, viz_tile_size, viz_tile_stride],
                        outputs=[grid_output, tile_count_output, tile_info_md]
                    )

                # Model information section
                self.create_model_info_accordion()

                # Tips section
                with gr.Accordion("💡 Usage Tips", open=False):
                    if self.args.model_variant == 'lightning':
                        tips = """
                        **⚡ Lightning Model Tips:**
                        - Use 4-8 steps for optimal speed/quality balance
                        - Lower CFG scales (1.5-3.0) work best
                        - Very fast generation, ideal for quick iterations
                        - Tiling can help with large images while maintaining speed
                        """
                    elif self.args.model_variant == 'tiled':
                        tips = """
                        **🏗️ Tiled Model Tips:**
                        - Optimized for large images and limited VRAM
                        - Use higher upscale factors (3x-4x) safely
                        - Enable "Tiled Processing" for maximum memory efficiency
                        - May take longer but handles 4K+ images smoothly
                        - Automatic tiled VAE enabled by default
                        """
                    else:
                        tips = """
                        **⚖️ Base Model Tips:**
                        - Balanced quality and speed
                        - Use 30-50 steps for best quality
                        - CFG scale 7-12 for most prompts
                        - Enable tiling for large images or low VRAM
                        - Good starting point for experimentation
                        """

                    tips += """

                    **🎛️ Tiling Guidelines:**
                    - **Low VRAM (≤8GB):** tile_size=96, stride=48, enable tiled VAE
                    - **Medium VRAM (8-16GB):** tile_size=128, stride=64
                    - **High VRAM (≥16GB):** tile_size=160, stride=80 or disable tiling

                    **🖼️ Image Size Recommendations:**
                    - **≤2K images:** Tiling usually not needed
                    - **2K-4K images:** Enable tiling for safety
                    - **≥4K images:** Always use tiling + tiled VAE

                    **🔄 Multi-Pass Processing:**
                    - **2 Passes:** Good for moderate enhancement
                    - **3 Passes:** Dramatic results for old/damaged photos
                    - **4 Passes:** Maximum enhancement for severely degraded images
                    """

                    gr.Markdown(tips)

            return interface

        self.interface = create_interface()

    def launch(self):
        """Launch the unified Gradio interface"""
        self.setup_gradio_interface()
        print(f"🚀 Launching {self.args.model_variant} variant demo...")
        print(f"🌐 Server: http://{self.args.ip}:{self.args.port}")
        print(f"🔧 Tiled Processing: {'Enabled' if self.enable_tiling else 'Available'}")

        self.interface.launch(
            server_name=self.args.ip,
            server_port=self.args.port,
            share=True,
            show_error=True,
            inbrowser=False
        )


def main():
    """Main entry point with enhanced argument parsing"""
    parser = create_base_parser()

    # Add demo-specific arguments
    parser.add_argument("--auto_optimize", action='store_true', default=False,
                        help="Automatically enable optimizations based on available hardware")

    args = parser.parse_args()

    # Auto-optimization logic
    if args.auto_optimize:
        print("🔧 Auto-optimization enabled...")

        # Detect VRAM and adjust settings
        if torch.cuda.is_available():
            try:
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
                print(f"📊 Detected {total_vram:.1f}GB VRAM")

                if total_vram <= 8:
                    print("🔧 Low VRAM detected: enabling aggressive optimizations")
                    args.enable_tiling = True
                    args.use_tile_vae = True
                    args.enable_cpu_offload = True
                    args.tile_size = 96
                    args.tile_stride = 48
                elif total_vram <= 16:
                    print("🔧 Medium VRAM detected: enabling balanced optimizations")
                    args.use_tile_vae = True
                    args.tile_size = 128
                    args.tile_stride = 64
                else:
                    print("🔧 High VRAM detected: minimal optimizations")
                    args.enable_xformers = True

            except Exception as e:
                print(f"⚠️ Could not detect VRAM: {e}")

    print("🚀 Starting PhotoAI Demo...")
    print(f"📋 Model variant: {args.model_variant}")
    print(f"📄 Config: {get_config_path(args.model_variant)}")

    if args.enable_tiling:
        print(f"🏗️ Tiled processing: {args.tile_size}x{args.tile_stride} (latent)")

    if args.use_tile_vae:
        print(f"🔄 Tiled VAE: {args.encoder_tile_size}x{args.decoder_tile_size}")

    # Create and launch demo
    try:
        demo = UnifiedPhotoAIDemo(args)
        demo.launch()
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Failed to start demo: {e}")
        raise


if __name__ == "__main__":
    main()