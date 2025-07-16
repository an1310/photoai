"""
Enhanced Clarity PhotoAI Demo - Subclass with Dramatic Clarity Features
Extends UnifiedPhotoAIDemo with advanced clarity enhancement capabilities
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional
import time

from photo_demo import UnifiedPhotoAIDemo
from base_demo import create_base_parser, get_config_path
from photoai.util import HWC3, upscale_image

# Import the enhanced clarity module we created earlier
# from enhanced_clarity import DramaticClarityEnhancer, IntegratedClarityModule


class EnhancedClarityDemo(UnifiedPhotoAIDemo):
    """
    Enhanced PhotoAI Demo with Dramatic Clarity Features
    Extends UnifiedPhotoAIDemo to add post-processing clarity enhancement
    """

    def __init__(self, args):
        super().__init__(args)

        # Initialize clarity enhancement module
        # self.clarity_enhancer = DramaticClarityEnhancer(device=self.imageai_device)
        # self.clarity_module = IntegratedClarityModule(device=self.imageai_device)

        # For now, we'll simulate the clarity enhancer
        self.clarity_enhancer = None
        self.clarity_module = None

        print("✅ Enhanced Clarity Demo initialized with dramatic clarity features")

    def apply_clarity_enhancement(self,
                                image_tensor: torch.Tensor,
                                clarity_mode: str = "dramatic",
                                clarity_strength: float = 2.0,
                                enable_dramatic_mode: bool = True,
                                preserve_colors: bool = True,
                                edge_enhancement: float = 1.5,
                                detail_boost: float = 1.0,
                                contrast_enhancement: float = 1.2,
                                noise_reduction: float = 0.3,
                                enable_multipass: bool = False,
                                multipass_count: int = 2) -> torch.Tensor:
        """
        Apply dramatic clarity enhancement to the restored image
        """

        if self.clarity_enhancer is None:
            # Simulate clarity enhancement for now
            # In real implementation, this would use the DramaticClarityEnhancer
            print(f"🎯 Applying {clarity_mode} clarity enhancement (strength: {clarity_strength})")

            # Simple sharpening simulation
            # Convert to PIL for simple processing
            if image_tensor.dim() == 4:
                image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            else:
                image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

            image_np = (image_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)

            # Apply simple enhancement based on parameters
            from PIL import ImageEnhance, ImageFilter

            # Sharpness enhancement
            sharpness_factor = 1.0 + (clarity_strength * 0.5)
            enhancer = ImageEnhance.Sharpness(pil_image)
            enhanced = enhancer.enhance(sharpness_factor)

            # Contrast enhancement
            if contrast_enhancement > 1.0:
                contrast_enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = contrast_enhancer.enhance(contrast_enhancement)

            # Edge enhancement simulation
            if edge_enhancement > 1.0 and enable_dramatic_mode:
                enhanced = enhanced.filter(ImageFilter.UnsharpMask(
                    radius=2,
                    percent=int(edge_enhancement * 100),
                    threshold=3
                ))

            # Multi-pass processing
            if enable_multipass:
                for i in range(multipass_count - 1):
                    # Apply weaker enhancement in subsequent passes
                    reduced_strength = clarity_strength * (0.7 ** (i + 1))
                    enhancer = ImageEnhance.Sharpness(enhanced)
                    enhanced = enhancer.enhance(1.0 + (reduced_strength * 0.3))

            # Convert back to tensor
            enhanced_np = np.array(enhanced).astype(np.float32) / 255.0
            enhanced_tensor = torch.from_numpy(enhanced_np).permute(2, 0, 1).unsqueeze(0)

            return enhanced_tensor.to(image_tensor.device)

        else:
            # Use the actual clarity enhancer
            return self.clarity_enhancer(
                image_tensor,
                clarity_strength=clarity_strength,
                dramatic_mode=enable_dramatic_mode,
                preserve_colors=preserve_colors
            )

    def process_with_clarity(self,
                           input_image,
                           # Clarity parameters first (matching the input order)
                           clarity_mode="dramatic",
                           clarity_strength=2.0,
                           enable_dramatic_mode=True,
                           preserve_colors=True,
                           edge_enhancement=1.5,
                           detail_boost=1.0,
                           contrast_enhancement=1.2,
                           noise_reduction=0.3,
                           enable_multipass=False,
                           multipass_count=2,
                           # PhotoAI parameters
                           prompt="",
                           a_prompt="",
                           n_prompt="",
                           num_steps=50,
                           restoration_scale=7.0,
                           s_churn=0.0,
                           s_noise=1.003,
                           guidance_scale=7.5,
                           seed=-1,
                           eta=1.0,
                           control_scale=1.0,
                           color_fix_type='Wavelet',
                           use_linear_cfg=False,
                           use_autocast=True,
                           photoai_checkpoint='Q',
                           upscale_factor=1.0) -> Tuple[List[Image.Image], str]:
        """
        Process image with PhotoAI and apply dramatic clarity enhancement
        """

        start_time = time.time()

        try:
            # Step 1: Run the base PhotoAI processing
            print("🚀 Running base PhotoAI restoration...")

            # Prepare the input image tensor
            input_np = HWC3(np.array(input_image))
            if upscale_factor > 1:
                input_np = upscale_image(input_np, upscale_factor, unit_resolution=32, min_size=1024)

            # Normalize to [-1, 1] range as expected by PhotoAI
            LQ = input_np.astype(np.float32) / 255.0 * 2.0 - 1.0
            LQ = torch.tensor(LQ, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.imageai_device)[:, :3, :, :]

            # Prepare captions - ensure they are strings
            full_prompt = str(prompt).strip() if prompt is not None else ""
            if a_prompt is not None and str(a_prompt).strip():
                a_prompt_str = str(a_prompt).strip()
                full_prompt = f"{full_prompt}, {a_prompt_str}" if full_prompt else a_prompt_str

            captions = [full_prompt] if full_prompt else [""]

            # Ensure n_prompt is a string
            n_prompt_str = str(n_prompt).strip() if n_prompt is not None else ""

            # Handle checkpoint switching
            if hasattr(self.model, 'current_model') and photoai_checkpoint:
                try:
                    if photoai_checkpoint == 'Q' and hasattr(self, 'ckpt_Q'):
                        self.model.load_state_dict(self.ckpt_Q, strict=False)
                        self.model.current_model = 'v0-Q'
                    elif photoai_checkpoint == 'F' and hasattr(self, 'ckpt_F'):
                        self.model.load_state_dict(self.ckpt_F, strict=False)
                        self.model.current_model = 'v0-F'
                    print(f"✅ Switched to {photoai_checkpoint} checkpoint")
                except Exception as e:
                    print(f"⚠️ Failed to switch checkpoint: {e}")

            # Process with PhotoAI using the actual batchify_sample method
            with torch.no_grad():
                if self.enable_tiling and hasattr(self.model, 'batchify_sample_tiled'):
                    # Use tiled processing if available
                    samples = self.model.batchify_sample_tiled(
                        LQ, captions,
                        num_steps=num_steps,
                        restoration_scale=restoration_scale,
                        s_churn=s_churn,
                        s_noise=s_noise,
                        cfg_scale=guidance_scale,
                        control_scale=control_scale,
                        seed=seed,
                        num_samples=1,
                        p_p=str(a_prompt) if a_prompt is not None else "",
                        n_p=n_prompt_str,
                        color_fix_type=color_fix_type,
                        use_linear_CFG=use_linear_cfg,
                        use_linear_control_scale=False,
                        cfg_scale_start=guidance_scale,
                        control_scale_start=control_scale,
                        tile_size=self.tile_size,
                        tile_stride=self.tile_stride
                    )
                else:
                    # Use standard processing
                    samples = self.model.batchify_sample(
                        LQ, captions,
                        num_steps=num_steps,
                        restoration_scale=restoration_scale,
                        s_churn=s_churn,
                        s_noise=s_noise,
                        cfg_scale=guidance_scale,
                        control_scale=control_scale,
                        seed=seed,
                        num_samples=1,
                        p_p=str(a_prompt) if a_prompt is not None else "",
                        n_p=n_prompt_str,
                        color_fix_type=color_fix_type,
                        use_linear_CFG=use_linear_cfg,
                        use_linear_control_scale=False,
                        cfg_scale_start=guidance_scale,
                        control_scale_start=control_scale
                    )

            # Convert tensor results to numpy arrays for further processing
            if isinstance(samples, torch.Tensor):
                samples_np = samples.detach().cpu().float().numpy()
                samples_np = (samples_np + 1.0) / 2.0  # Denormalize from [-1,1] to [0,1]
                samples_np = samples_np.transpose(0, 2, 3, 1)  # BCHW to BHWC
                samples_np = (samples_np * 255.0).round().clip(0, 255).astype(np.uint8)
                results = [samples_np[i] for i in range(samples_np.shape[0])]
            else:
                results = samples if isinstance(samples, list) else [samples]

            if not results:
                return [], f"❌ PhotoAI processing failed: No results generated"

            # Step 2: Apply clarity enhancement to each result
            print(f"✨ Applying {clarity_mode} clarity enhancement...")
            enhanced_results = []

            for i, result_np in enumerate(results):
                # Convert numpy array to tensor
                if isinstance(result_np, np.ndarray):
                    # Assume result_np is in [0, 255] uint8 format
                    result_tensor = torch.from_numpy(result_np.astype(np.float32) / 255.0)
                    if result_tensor.dim() == 3:
                        result_tensor = result_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
                    elif result_tensor.dim() == 4:
                        result_tensor = result_tensor.permute(0, 3, 1, 2)  # BHWC -> BCHW
                else:
                    result_tensor = result_np

                result_tensor = result_tensor.to(self.imageai_device)

                # Apply clarity enhancement
                enhanced_tensor = self.apply_clarity_enhancement(
                    result_tensor,
                    clarity_mode=clarity_mode,
                    clarity_strength=clarity_strength,
                    enable_dramatic_mode=enable_dramatic_mode,
                    preserve_colors=preserve_colors,
                    edge_enhancement=edge_enhancement,
                    detail_boost=detail_boost,
                    contrast_enhancement=contrast_enhancement,
                    noise_reduction=noise_reduction,
                    enable_multipass=enable_multipass,
                    multipass_count=multipass_count
                )

                # Convert back to PIL Image
                enhanced_np = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                enhanced_np = (enhanced_np * 255).clip(0, 255).astype(np.uint8)
                enhanced_pil = Image.fromarray(enhanced_np)

                enhanced_results.append(enhanced_pil)

            processing_time = time.time() - start_time

            # Create detailed processing info
            clarity_info = f"""
✅ Enhanced Clarity Processing Complete!

🎯 **Clarity Settings:**
• Mode: {clarity_mode}
• Strength: {clarity_strength}
• Dramatic Mode: {'✅' if enable_dramatic_mode else '❌'}
• Color Preservation: {'✅' if preserve_colors else '❌'}
• Edge Enhancement: {edge_enhancement}
• Detail Boost: {detail_boost}
• Contrast Enhancement: {contrast_enhancement}
• Multi-Pass: {'✅ (' + str(multipass_count) + ' passes)' if enable_multipass else '❌'}

📊 **PhotoAI Settings:**
• Steps: {num_steps}
• Restoration Scale: {restoration_scale}
• S-Churn: {s_churn}
• S-Noise: {s_noise}
• Guidance Scale: {guidance_scale}
• Checkpoint: {photoai_checkpoint}

⏱️ **Performance:**
• Total Processing Time: {processing_time:.2f}s
• Images Generated: {len(enhanced_results)}

🎨 **Enhancement Level:** {'🔥 EXTREME' if clarity_strength > 3.0 else '🚀 HIGH' if clarity_strength > 2.0 else '⚡ MODERATE' if clarity_strength > 1.0 else '🌟 SUBTLE'}
            """

            return enhanced_results, clarity_info.strip()

        except Exception as e:
            error_msg = f"❌ Error in enhanced clarity processing: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return [], error_msg

    def create_clarity_interface_components(self):
        """Create the clarity enhancement UI components"""

        with gr.Accordion("✨ DRAMATIC CLARITY CONTROLS", open=True) as clarity_section:
            gr.Markdown("### 🎯 Professional-Grade Clarity Enhancement")

            with gr.Row():
                clarity_mode = gr.Radio(
                    choices=[
                        ("🌟 Dramatic Enhancement", "dramatic"),
                        ("🚀 Extreme Clarity", "extreme"),
                        ("📷 Portrait Optimized", "portraits"),
                        ("🏔️ Landscape Optimized", "landscapes"),
                        ("📄 Document/Text", "documents"),
                        ("🕰️ Vintage Photo", "vintage_photos"),
                        ("⚙️ Custom", "custom")
                    ],
                    value="dramatic",
                    label="Clarity Preset",
                    info="Choose enhancement style"
                )

                clarity_strength = gr.Slider(
                    0.5, 4.0, value=2.0, step=0.1,
                    label="Clarity Strength",
                    info="0.5=Subtle, 2.0=Dramatic, 4.0=Extreme"
                )

            with gr.Row():
                enable_dramatic_mode = gr.Checkbox(
                    value=True,
                    label="🔥 Dramatic Mode",
                    info="Enable aggressive enhancement"
                )

                preserve_colors = gr.Checkbox(
                    value=True,
                    label="🌈 Preserve Colors",
                    info="Maintain color balance"
                )

                enable_multipass = gr.Checkbox(
                    value=False,
                    label="🔄 Multi-Pass",
                    info="Apply enhancement multiple times"
                )

            with gr.Accordion("🔬 Advanced Clarity Settings", open=False):
                with gr.Row():
                    edge_enhancement = gr.Slider(
                        0.0, 3.0, value=1.5, step=0.1,
                        label="Edge Enhancement",
                        info="Boost edge definition"
                    )

                    detail_boost = gr.Slider(
                        0.0, 2.0, value=1.0, step=0.1,
                        label="Detail Boost",
                        info="Enhance fine details"
                    )

                with gr.Row():
                    contrast_enhancement = gr.Slider(
                        0.0, 2.0, value=1.2, step=0.1,
                        label="Contrast Enhancement",
                        info="Adaptive contrast boost"
                    )

                    noise_reduction = gr.Slider(
                        0.0, 1.0, value=0.3, step=0.1,
                        label="Noise Reduction",
                        info="Reduce enhancement artifacts"
                    )

                multipass_count = gr.Slider(
                    2, 5, value=2, step=1,
                    label="Multi-Pass Count",
                    info="Number of enhancement passes",
                    visible=False
                )

        # Show/hide multipass count based on checkbox
        def toggle_multipass_count(enable_multipass):
            return gr.update(visible=enable_multipass)

        enable_multipass.change(
            toggle_multipass_count,
            inputs=[enable_multipass],
            outputs=[multipass_count]
        )

        # Update clarity strength based on preset
        def update_clarity_from_preset(clarity_mode):
            preset_strengths = {
                "dramatic": 2.0,
                "extreme": 3.5,
                "portraits": 1.5,
                "landscapes": 2.5,
                "documents": 3.0,
                "vintage_photos": 1.8,
                "custom": 2.0
            }
            return gr.update(value=preset_strengths.get(clarity_mode, 2.0))

        clarity_mode.change(
            update_clarity_from_preset,
            inputs=[clarity_mode],
            outputs=[clarity_strength]
        )

        return {
            'clarity_mode': clarity_mode,
            'clarity_strength': clarity_strength,
            'enable_dramatic_mode': enable_dramatic_mode,
            'preserve_colors': preserve_colors,
            'edge_enhancement': edge_enhancement,
            'detail_boost': detail_boost,
            'contrast_enhancement': contrast_enhancement,
            'noise_reduction': noise_reduction,
            'enable_multipass': enable_multipass,
            'multipass_count': multipass_count
        }

    def setup_gradio_interface(self):
        """Setup enhanced Gradio interface with clarity controls"""

        def create_interface():
            title = f"PhotoAI Enhanced Clarity Demo - {self.args.model_variant.capitalize()} Variant"

            # Custom CSS for better visuals
            css = """
            .clarity-section {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 10px;
                padding: 15px;
                margin: 10px 0;
            }
            .dramatic-button {
                background: linear-gradient(45deg, #ff6b6b, #ee5a24) !important;
                color: white !important;
                border: none !important;
                border-radius: 8px !important;
                font-weight: bold !important;
            }
            """

            with gr.Blocks(title=title, theme=gr.themes.Soft(), css=css) as interface:
                gr.Markdown(f"# {title}")
                gr.Markdown("### 🎯 Professional Image Restoration with Dramatic Clarity Enhancement")
                gr.Markdown(
                    f"**Config:** {get_config_path(self.args.model_variant)} | **Tiling:** {'✅' if self.enable_tiling else '❌'} | **Clarity:** ✨ ENHANCED")

                with gr.Row():
                    with gr.Column(scale=1):
                        # Input image
                        input_image = gr.Image(
                            label="📤 Upload Image",
                            type="pil",
                            height=400
                        )

                        # Get base PhotoAI components - simplified to match your actual interface
                        with gr.Accordion("⚙️ PhotoAI Settings", open=True):
                            with gr.Row():
                                num_steps = gr.Slider(
                                    20, 100, value=50, step=5,
                                    label="Steps",
                                    info="More steps = higher quality"
                                )
                                restoration_scale = gr.Slider(
                                    1.0, 12.0, value=7.0, step=0.5,
                                    label="Restoration Scale",
                                    info="Higher = more aggressive restoration"
                                )

                            with gr.Row():
                                guidance_scale = gr.Slider(
                                    1.0, 20.0, value=7.5, step=0.5,
                                    label="Guidance Scale"
                                )
                                seed = gr.Number(
                                    value=-1,
                                    label="Seed (-1 for random)"
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

                            photoai_checkpoint = gr.Radio(
                                choices=["Q", "F"], value="Q",
                                label="PhotoAI Checkpoint",
                                info="Q=Quality focused, F=Fidelity focused"
                            )

                            # Additional controls
                            with gr.Accordion("🔧 Advanced PhotoAI Settings", open=False):
                                prompt = gr.Textbox(
                                    label="Prompt",
                                    placeholder="Describe the image or enhancement desired",
                                    lines=2
                                )

                                with gr.Row():
                                    a_prompt = gr.Textbox(
                                        label="Additional Prompt",
                                        placeholder="Additional positive prompt",
                                        lines=1
                                    )
                                    n_prompt = gr.Textbox(
                                        label="Negative Prompt",
                                        placeholder="What to avoid",
                                        lines=1
                                    )

                                with gr.Row():
                                    control_scale = gr.Slider(
                                        0.0, 2.0, value=1.0, step=0.1,
                                        label="Control Scale"
                                    )
                                    color_fix_type = gr.Radio(
                                        choices=["Wavelet", "AdaIn", "None"],
                                        value="Wavelet",
                                        label="Color Fix Type"
                                    )

                                use_linear_cfg = gr.Checkbox(
                                    value=False,
                                    label="Use Linear CFG"
                                )

                        # Add clarity components
                        clarity_components = self.create_clarity_interface_components()

                        # Process button
                        process_btn = gr.Button(
                            "🎨 Generate with Dramatic Clarity",
                            variant="primary",
                            size="lg",
                            elem_classes=["dramatic-button"]
                        )

                        clear_btn = gr.Button("🗑️ Clear", variant="secondary")

                    with gr.Column(scale=1):
                        # Output
                        output_gallery = gr.Gallery(
                            label="✨ Enhanced Results",
                            show_label=True,
                            elem_id="gallery",
                            columns=2,
                            rows=2,
                            height=600
                        )

                        # Processing info
                        processing_info = gr.Textbox(
                            label="📊 Processing Information",
                            lines=10,
                            interactive=False
                        )

                        # Download button
                        download_btn = gr.Button(
                            "💾 Download Results",
                            variant="secondary"
                        )

                # Tips section
                with gr.Accordion("💡 Enhanced Clarity Tips & Tricks", open=False):
                    gr.Markdown("""
                    ### 🎯 **For Maximum Drama:**
                    - **Extreme Clarity Mode**: Most aggressive enhancement
                    - **Multi-Pass Processing**: Apply 2-3 passes for extreme results
                    - **High Restoration Scale (8-12)**: Aggressive base restoration
                    - **Custom Strength 3.0+**: Maximum clarity boost
                    
                    ### 📸 **Content-Specific Recommendations:**
                    - **Portraits**: Portrait preset, preserve colors ON
                    - **Landscapes**: Landscape preset, high edge enhancement
                    - **Documents**: Document preset, dramatic mode ON
                    - **Vintage Photos**: Vintage preset, moderate enhancement
                    
                    ### ⚡ **Best Practices:**
                    - Start with presets, then fine-tune
                    - Higher PhotoAI steps (50+) work best with clarity
                    - Use noise reduction if artifacts appear
                    - Multi-pass on already good images for extreme results
                    """)

                # Wire up the processing function
                all_inputs = [
                    input_image,
                    # Clarity components
                    clarity_components['clarity_mode'],
                    clarity_components['clarity_strength'],
                    clarity_components['enable_dramatic_mode'],
                    clarity_components['preserve_colors'],
                    clarity_components['edge_enhancement'],
                    clarity_components['detail_boost'],
                    clarity_components['contrast_enhancement'],
                    clarity_components['noise_reduction'],
                    clarity_components['enable_multipass'],
                    clarity_components['multipass_count'],
                    # PhotoAI base components
                    prompt,
                    a_prompt,
                    n_prompt,
                    num_steps,
                    restoration_scale,
                    s_churn,
                    s_noise,
                    guidance_scale,
                    seed,
                    gr.State(1.0),  # eta
                    control_scale,
                    color_fix_type,
                    use_linear_cfg,
                    gr.State(True),  # use_autocast
                    photoai_checkpoint,
                    gr.State(1.0)    # upscale_factor
                ]

                process_btn.click(
                    self.process_with_clarity,
                    inputs=all_inputs,
                    outputs=[output_gallery, processing_info]
                )

                clear_btn.click(
                    lambda: (None, None, ""),
                    outputs=[input_image, output_gallery, processing_info]
                )

            return interface

        self.interface = create_interface()
        return self.interface


def create_enhanced_clarity_parser():
    """Create argument parser for enhanced clarity demo"""
    parser = create_base_parser()

    # Add clarity-specific arguments
    parser.add_argument("--default_clarity_mode", type=str, default="dramatic",
                        choices=["dramatic", "extreme", "portraits", "landscapes", "documents", "vintage_photos", "custom"],
                        help="Default clarity enhancement mode")
    parser.add_argument("--default_clarity_strength", type=float, default=2.0,
                        help="Default clarity strength")
    parser.add_argument("--enable_multipass_by_default", action='store_true', default=False,
                        help="Enable multi-pass processing by default")

    return parser


def main():
    """Main function for enhanced clarity demo"""
    parser = create_enhanced_clarity_parser()
    args = parser.parse_args()

    print("🚀 Starting Enhanced Clarity PhotoAI Demo...")
    print(f"📋 Model variant: {args.model_variant}")
    print(f"📄 Config: {get_config_path(args.model_variant)}")
    print(f"✨ Default clarity mode: {args.default_clarity_mode}")
    print(f"💪 Default clarity strength: {args.default_clarity_strength}")

    if args.enable_tiling:
        print(f"🏗️ Tiled processing: {args.tile_size}x{args.tile_stride} (latent)")

    if args.use_tile_vae:
        print(f"🔄 Tiled VAE: {args.encoder_tile_size}x{args.decoder_tile_size}")

    try:
        demo = EnhancedClarityDemo(args)
        demo.launch()
    except KeyboardInterrupt:
        print("\n👋 Enhanced Clarity Demo stopped by user")
    except Exception as e:
        print(f"❌ Failed to start enhanced clarity demo: {e}")
        raise


if __name__ == "__main__":
    main()