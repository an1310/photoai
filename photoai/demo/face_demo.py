"""
Face-Enhanced PhotoAI Demo with Advanced Controls
Combines face detection/alignment with the advanced PhotoAI parameters
"""
import gradio as gr
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any
import copy
import time

from base_demo import ImageAIDemoBase, create_base_parser, get_config_path, setup_model_optimizations
from photoai.util import HWC3, upscale_image, convert_dtype, create_photoai_model, load_QF_ckpt
from photoai.utils.face_restoration_helper import FaceRestoreHelper


class FaceEnhancedDemo(ImageAIDemoBase):
    """Face-optimized PhotoAI demo with advanced enhancement controls"""

    def __init__(self, args):
        super().__init__(args)
        self.interface = None

        # Initialize face helper
        self.face_helper = FaceRestoreHelper(
            device=self.imageai_device,
            upscale_factor=1,
            face_size=1024,
            use_parse=True,
            det_model='retinaface_resnet50'
        )

        print("✅ Face detection helper initialized")

    def setup_imageai_model(self):
        """Setup PhotoAI model for face processing"""
        print(f"Loading PhotoAI model for face enhancement: {self.args.model_variant}")

        config_path = get_config_path(self.args.model_variant)
        print(f"Using config: {config_path}")

        try:
            # Create PhotoAI model
            self.model = create_photoai_model(
                config_path=config_path,
                photoai_sign='Q',
                load_default_setting=False
            )

            # Apply optimizations
            self.model = setup_model_optimizations(self.model, self.args)

            # Move to device
            self.model = self.model.to(self.imageai_device)

            # Setup denoise encoder for stage 1
            if hasattr(self.model.first_stage_model, 'denoise_encoder'):
                self.model.first_stage_model.denoise_encoder_s1 = copy.deepcopy(
                    self.model.first_stage_model.denoise_encoder
                )

            # Set current model and load checkpoints
            self.model.current_model = 'v0-Q'
            self.ckpt_Q, self.ckpt_F = load_QF_ckpt(config_path)

            print("✅ PhotoAI face model loaded successfully!")

            # Store model info
            model_info = self.get_model_info()
            print(f"Model type: {model_info['model_type']}")
            print(f"VAE: {model_info['vae_type']}")

        except Exception as e:
            print(f"❌ Failed to load PhotoAI face model: {e}")
            raise

    def face_detect_and_analyze(self, input_image):
        """Detect faces and return analysis info"""
        if input_image is None:
            return "No image provided", [], 0

        try:
            # Clean previous faces
            self.face_helper.clean_all()

            # Detect faces
            self.face_helper.read_image(input_image)
            self.face_helper.get_face_landmarks_5(
                only_center_face=False,
                resize=640,
                eye_dist_threshold=5
            )

            num_faces = len(self.face_helper.all_landmarks_5)

            if num_faces == 0:
                return "No faces detected in image", [], 0

            # Align and extract faces
            self.face_helper.align_warp_face()

            # Convert cropped faces to PIL Images
            face_images = []
            for i, face in enumerate(self.face_helper.cropped_faces):
                face_pil = Image.fromarray(face)
                face_images.append(face_pil)

            analysis = f"✅ Detected {num_faces} face{'s' if num_faces != 1 else ''}"

            return analysis, face_images, num_faces

        except Exception as e:
            return f"❌ Face detection error: {str(e)}", [], 0

    def process_face_enhanced(self, input_image, prompt, a_prompt, n_prompt,
                              upscale_factor, num_steps, guidance_scale, control_scale, seed,
                              color_fix_type, restoration_scale=4.0, s_churn=0.0, s_noise=1.003,
                              photoai_checkpoint="Q", apply_face_enhancement=True, face_resolution=1024):
        """Process with face-specific enhancements"""

        if input_image is None:
            return [], "No input image provided", ""

        # Handle checkpoint switching
        if hasattr(self.model, 'current_model') and self.model.current_model != f'v0-{photoai_checkpoint}':
            try:
                if photoai_checkpoint == 'Q':
                    self.model.load_state_dict(self.ckpt_Q, strict=False)
                    self.model.current_model = 'v0-Q'
                elif photoai_checkpoint == 'F':
                    self.model.load_state_dict(self.ckpt_F, strict=False)
                    self.model.current_model = 'v0-F'
                print(f"✅ Switched to {photoai_checkpoint} checkpoint")
            except Exception as e:
                print(f"⚠️ Failed to switch checkpoint: {e}")

        try:
            torch.cuda.set_device(self.imageai_device)

            # Prepare input image
            input_image = HWC3(input_image)
            if upscale_factor > 1:
                input_image = upscale_image(input_image, upscale_factor, unit_resolution=32, min_size=1024)

            # Face detection and analysis
            if apply_face_enhancement:
                face_analysis, face_images, num_faces = self.face_detect_and_analyze(input_image)
                processing_info = f"Face Enhancement: {face_analysis}\n"
            else:
                processing_info = "Face Enhancement: Disabled\n"

            # Prepare image tensor
            LQ = np.array(input_image) / 255.0 * 2.0 - 1.0
            LQ = torch.tensor(LQ, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.imageai_device)[:, :3, :,
                 :]

            # Prepare captions
            full_prompt = prompt.strip()
            if a_prompt.strip():
                full_prompt = f"{full_prompt}, {a_prompt.strip()}" if full_prompt else a_prompt.strip()

            captions = [full_prompt] if full_prompt else [""]

            # Process with PhotoAI
            with torch.no_grad():
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
                    p_p=a_prompt,
                    n_p=n_prompt,
                    color_fix_type=color_fix_type,
                    use_linear_CFG=False,
                    use_linear_control_scale=False,
                    cfg_scale_start=guidance_scale,
                    control_scale_start=control_scale
                )

            # Convert results
            if isinstance(samples, torch.Tensor):
                samples_np = samples.detach().cpu().float().numpy()
                samples_np = (samples_np + 1.0) / 2.0
                samples_np = samples_np.transpose(0, 2, 3, 1)
                samples_np = (samples_np * 255.0).round().clip(0, 255).astype(np.uint8)
                results = [samples_np[i] for i in range(samples_np.shape[0])]
            else:
                results = samples

            # Create event ID for logging
            event_id = str(time.time_ns())

            processing_info += f"Enhancement completed - Event ID: {event_id}"

            return results, event_id, processing_info

        except Exception as e:
            error_msg = f"Error in face processing: {str(e)}"
            print(error_msg)
            return [], error_msg, error_msg

    def setup_gradio_interface(self):
        """Setup face-optimized Gradio interface"""

        def create_interface():
            with gr.Blocks(title="PhotoAI Face Enhancement", theme=gr.themes.Soft()) as interface:
                gr.Markdown("# 🎭 PhotoAI Face Enhancement Demo")
                gr.Markdown("**Specialized for portrait and family photo restoration with advanced face detection**")

                with gr.Tab("👨‍👩‍👧‍👦 Face Enhancement"):
                    with gr.Row():
                        with gr.Column():
                            input_image = gr.Image(label="Input Image", type="numpy")

                            # Face-specific settings
                            with gr.Accordion("👤 Face Processing Settings", open=True):
                                apply_face_enhancement = gr.Checkbox(
                                    value=True,
                                    label="Enable Face Detection & Analysis",
                                    info="Detect and analyze faces for optimized enhancement"
                                )
                                face_resolution = gr.Slider(
                                    512, 2048, value=1024, step=64,
                                    label="Face Resolution",
                                    info="Target resolution for face processing"
                                )

                            # Get common components
                            components = self.create_common_interface_components()

                            # Face-optimized prompts
                            with gr.Accordion("🎨 Face Enhancement Prompts", open=False):
                                prompt = gr.Textbox(
                                    label="Enhancement Prompt",
                                    value="professional portrait, natural skin texture, sharp facial features",
                                    lines=2
                                )
                                a_prompt = gr.Textbox(
                                    label="Additional Prompt",
                                    value="detailed eyes, realistic skin, professional lighting, crystal clear, high quality",
                                    lines=2
                                )
                                n_prompt = gr.Textbox(
                                    label="Negative Prompt",
                                    value="plastic skin, oversmoothed, artificial, soft focus, blurry eyes, low quality",
                                    lines=2
                                )

                            # Advanced settings (merged from base)
                            with gr.Accordion("Advanced Settings", open=False):
                                with gr.Row():
                                    num_samples = gr.Slider(1, 4, value=1, step=1, label="Number of Samples")
                                    use_linear_cfg = gr.Checkbox(value=False, label="Linear CFG")

                                gr.Markdown("---")
                                gr.Markdown("**PhotoAI Enhancement Parameters**")

                                with gr.Row():
                                    restoration_scale = gr.Slider(
                                        1.0, 8.0, value=5.0, step=0.1,  # Default 5.0 for faces
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
                                        0.0, 2.0, value=0.8, step=0.1,  # Default 0.8 for faces
                                        label="S-Churn",
                                        info="Stochasticity for detail generation"
                                    )
                                    s_noise = gr.Slider(
                                        0.5, 2.0, value=1.3, step=0.001,  # Default 1.3 for faces
                                        label="S-Noise",
                                        info="Noise injection for texture detail"
                                    )

                                with gr.Accordion("💡 Face Enhancement Tips", open=False):
                                    gr.Markdown("""
                                    **For Portrait Photos:**
                                    - **Restoration Scale 4-6:** Good for family photos
                                    - **Restoration Scale 6-8:** Dramatic enhancement for old/damaged photos
                                    - **S-Churn 0.5-1.0:** Adds natural skin texture variation
                                    - **Q Checkpoint:** Better for overall portrait quality
                                    - **F Checkpoint:** Better for preserving original facial features

                                    **Memory Tips:**
                                    - Enable tiling for large family group photos
                                    - Use face detection for automatic optimization
                                    """)

                            # Process button
                            process_btn = gr.Button("🎭 Enhance Faces", variant="primary", size="lg")

                        with gr.Column():
                            # Outputs
                            result_gallery = gr.Gallery(
                                label="Enhanced Results",
                                columns=1,
                                rows=1,
                                height=500
                            )
                            event_id_output = gr.Textbox(label="Event ID", interactive=False)
                            processing_info = gr.Textbox(
                                label="Processing Information",
                                lines=4,
                                interactive=False
                            )

                    # Face detection preview
                    with gr.Row():
                        with gr.Column():
                            detect_btn = gr.Button("🔍 Analyze Faces", variant="secondary")
                        with gr.Column():
                            face_analysis = gr.Textbox(label="Face Analysis", interactive=False)
                            detected_faces = gr.Gallery(
                                label="Detected Faces",
                                columns=4,
                                rows=1,
                                height=200
                            )

                    # Button handlers
                    process_btn.click(
                        self.process_face_enhanced,
                        inputs=[
                            input_image, prompt, a_prompt, n_prompt,
                            components['upscale_factor'], components['num_steps'],
                            components['guidance_scale'], components['control_scale'], components['seed'],
                            components['color_fix_type'], restoration_scale, s_churn, s_noise,
                            photoai_checkpoint, apply_face_enhancement, face_resolution
                        ],
                        outputs=[result_gallery, event_id_output, processing_info]
                    )

                    detect_btn.click(
                        self.face_detect_and_analyze,
                        inputs=[input_image],
                        outputs=[face_analysis, detected_faces, gr.State()]
                    )

                with gr.Tab("🔧 Stage 1 - Preprocessing"):
                    with gr.Row():
                        with gr.Column():
                            stage1_input = gr.Image(label="Input Image", type="numpy")
                            gamma_correction = gr.Slider(0.5, 2.0, value=1.0, label="Gamma Correction")
                            stage1_btn = gr.Button("Preprocess Image", variant="primary")

                        with gr.Column():
                            stage1_output = gr.Image(label="Preprocessed Image", type="numpy")

                    stage1_btn.click(
                        self.stage1_process,
                        inputs=[stage1_input, gamma_correction],
                        outputs=[stage1_output]
                    )

                # Model information
                self.create_model_info_accordion()

            return interface

        self.interface = create_interface()

    def launch(self):
        """Launch the face-enhanced demo"""
        self.setup_gradio_interface()
        print("🎭 Launching Face-Enhanced PhotoAI Demo...")
        print(f"🌐 Server: http://{self.args.ip}:{self.args.port}")
        print(f"👤 Face detection: Enabled")

        self.interface.launch(
            server_name=self.args.ip,
            server_port=self.args.port,
            share=True,
            show_error=True,
            inbrowser=False
        )


def main():
    """Main entry point for face-enhanced demo"""
    parser = create_base_parser()
    parser.add_argument("--face_resolution", type=int, default=1024,
                        help="Target resolution for face processing")

    args = parser.parse_args()

    print("🎭 Starting Face-Enhanced PhotoAI Demo...")
    print("🎯 Optimized for portrait and family photo restoration")

    try:
        demo = FaceEnhancedDemo(args)
        demo.launch()
    except KeyboardInterrupt:
        print("\n👋 Face demo stopped")
    except Exception as e:
        print(f"❌ Failed to start face demo: {e}")
        raise


if __name__ == "__main__":
    main()