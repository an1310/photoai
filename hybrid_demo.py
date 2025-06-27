#!/usr/bin/env python3
"""
Hybrid Gradio Demo: Modern UI from imageai/update + Working photoai functions
"""
import os
import logging
import traceback
import sys
import argparse
import time
import copy
from typing import Dict, Any, Optional, List

import gradio as gr
import torch
import numpy as np
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

# Import working photoai functions
from photoai.util import HWC3, upscale_image, fix_resize, convert_dtype, create_photoai_model, load_QF_ckpt
from CKPT_PTH import LLAVA_MODEL_PATH

# Setup logging BEFORE importing patch
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import and apply OpenCLIP patches - this must be done EARLY
try:
    from openclip_patch import apply_all_openclip_patches

    logger.info("OpenCLIP patch module imported successfully")
except ImportError as e:
    logger.error(f"Failed to import openclip_patch: {e}")
    logger.error("Make sure openclip_patch.py is in the same directory as this script")
    sys.exit(1)


class HybridImageAIDemo:
    """Hybrid demo using modern UI structure with working photoai functions"""

    def __init__(self, args):
        self.args = args
        self.model = None
        self.default_setting = None
        self.ckpt_Q = None
        self.ckpt_F = None
        self.llava_model = None
        self.llava_processor = None
        self.interface = None

        # Setup devices
        self.setup_devices()
        # Initialize models
        self.setup_photoai_model()
        self.setup_llava_model()

    def setup_devices(self):
        """Setup CUDA devices"""
        if torch.cuda.device_count() >= 2:
            self.photoai_device = 'cuda:0'
            self.llava_device = 'cuda:1'
        elif torch.cuda.device_count() == 1:
            self.photoai_device = 'cuda:0'
            self.llava_device = 'cuda:0'
        else:
            raise ValueError('Currently support CUDA only.')

        logger.info(f"PhotoAI device: {self.photoai_device}")
        logger.info(f"LLaVA device: {self.llava_device}")

    def setup_photoai_model(self):
        """Setup PhotoAI model using existing working code"""
        try:
            logger.info("Loading PhotoAI model...")

            # Apply OpenCLIP patches BEFORE loading any models
            logger.info("Applying OpenCLIP compatibility patches...")
            if not apply_all_openclip_patches():
                logger.warning("Some OpenCLIP patches failed to apply, but continuing...")

            # Use the working photoai model loading
            self.model, self.default_setting = create_photoai_model(
                self.args.opt,
                photoai_sign='Q',
                load_default_setting=True
            )

            if self.args.loading_half_params:
                logger.info("Converting model to half precision")
                self.model = self.model.half()

            if self.args.use_tile_vae:
                logger.info(f"Initializing tile VAE")
                self.model.init_tile_vae(
                    encoder_tile_size=self.args.encoder_tile_size,
                    decoder_tile_size=self.args.decoder_tile_size
                )

            self.model = self.model.to(self.photoai_device)
            self.model.first_stage_model.denoise_encoder_s1 = copy.deepcopy(
                self.model.first_stage_model.denoise_encoder
            )
            self.model.current_model = 'v0-Q'

            # Load checkpoints
            self.ckpt_Q, self.ckpt_F = load_QF_ckpt(self.args.opt)

            logger.info("✅ PhotoAI model loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load PhotoAI model: {e}")
            logger.error(traceback.format_exc())
            raise

    def setup_llava_model(self):
        """Setup LLaVA model using modern transformers approach"""
        self.use_llava = not self.args.no_llava

        if not self.use_llava:
            logger.info("LLaVA disabled by user")
            return

        try:
            logger.info("Loading LLaVA model using Transformers...")
            logger.info(f"LLaVA model path: {LLAVA_MODEL_PATH}")

            self.llava_processor = LlavaNextProcessor.from_pretrained(LLAVA_MODEL_PATH)

            if self.args.load_8bit_llava:
                logger.info("Loading LLaVA model in 8-bit mode...")
                self.llava_model = LlavaNextForConditionalGeneration.from_pretrained(
                    LLAVA_MODEL_PATH,
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    device_map=self.llava_device
                )
            else:
                logger.info("Loading LLaVA model in 16-bit mode...")
                self.llava_model = LlavaNextForConditionalGeneration.from_pretrained(
                    LLAVA_MODEL_PATH,
                    torch_dtype=torch.float16,
                    device_map=self.llava_device
                )

            self.llava_model.eval()
            logger.info("✅ LLaVA model loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load LLaVA model: {e}")
            logger.error(traceback.format_exc())
            logger.warning("Continuing without LLaVA")
            self.use_llava = False
            self.llava_model = None
            self.llava_processor = None

    def generate_llava_caption(self, image, temperature=0.2, top_p=0.9, qs=None):
        """Generate image caption using Transformers LLaVA"""
        if not self.use_llava or self.llava_model is None:
            return "LLaVA is not available. Please add text manually."

        try:
            if qs is None:
                qs = "Describe this image in detail."

            # Prepare conversation format
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": qs},
                        {"type": "image"},
                    ],
                },
            ]

            # Apply chat template and process
            prompt = self.llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.llava_processor(text=prompt, images=image, return_tensors="pt").to(self.llava_device)

            # Generate caption
            with torch.no_grad():
                output = self.llava_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.llava_processor.tokenizer.eos_token_id
                )

            # Decode and extract caption
            generated_text = self.llava_processor.decode(output[0], skip_special_tokens=True)

            if "assistant\n" in generated_text:
                caption = generated_text.split("assistant\n")[-1].strip()
            else:
                caption = generated_text.strip()

            return caption

        except Exception as e:
            logger.error(f"Error generating LLaVA caption: {e}")
            return "Error generating caption. Please add text manually."

    # Working PhotoAI functions adapted from original code
    def stage1_process(self, input_image, gamma_correction):
        """Stage 1 processing using working photoai functions"""
        logger.info("Starting Stage1 processing...")

        try:
            torch.cuda.set_device(self.photoai_device)

            # Convert PIL Image to numpy array for HWC3
            if hasattr(input_image, 'convert'):  # PIL Image
                input_image = input_image.convert('RGB')
                LQ = np.array(input_image).astype(np.uint8)
            else:  # Already numpy array
                LQ = input_image.astype(np.uint8)

            LQ = HWC3(LQ)
            LQ = fix_resize(LQ, 512)

            # Stage1 processing
            LQ = np.array(LQ) / 255 * 2 - 1
            LQ = torch.tensor(LQ, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.photoai_device)[:, :3, :,
                 :]
            LQ = self.model.batchify_denoise(LQ, is_stage1=True)
            LQ = (LQ[0].permute(1, 2, 0) * 127.5 + 127.5).cpu().numpy().round().clip(0, 255).astype(np.uint8)

            # Gamma correction
            LQ = LQ / 255.0
            LQ = np.power(LQ, gamma_correction)
            LQ *= 255.0
            LQ = LQ.round().clip(0, 255).astype(np.uint8)

            logger.info("Stage1 processing completed successfully")
            return LQ

        except Exception as e:
            logger.error(f"Error in stage1_process: {e}")
            logger.error(traceback.format_exc())
            raise

    def llava_process(self, input_image, temperature, top_p, qs=None):
        """LLaVA processing using working functions"""
        logger.info("Starting LLaVA processing...")

        try:
            torch.cuda.set_device(self.llava_device)

            if self.use_llava:
                # Convert for HWC3, then back to PIL for LLaVA
                if hasattr(input_image, 'convert'):  # PIL Image
                    input_image = input_image.convert('RGB')
                    LQ = np.array(input_image).astype(np.uint8)
                else:  # Already numpy array
                    LQ = input_image.astype(np.uint8)

                LQ = HWC3(LQ)
                LQ = Image.fromarray(LQ)

                caption = self.generate_llava_caption(LQ, temperature=temperature, top_p=top_p, qs=qs)
                logger.info(f"LLaVA processing completed: {caption}")
            else:
                caption = 'LLaVA is not available. Please add text manually.'
                logger.warning("LLaVA not available")

            return caption

        except Exception as e:
            logger.error(f"Error in llava_process: {e}")
            return f"Error in LLaVA processing: {str(e)}"

    def stage2_process(self, input_image, prompt, a_prompt, n_prompt, num_samples, upscale,
                       edm_steps, s_stage1, s_stage2, s_cfg, seed, s_churn, s_noise,
                       color_fix_type, diff_dtype, ae_dtype, gamma_correction,
                       linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2, model_select):
        """Stage 2 processing using working photoai functions"""
        logger.info("Starting Stage2 processing...")

        try:
            torch.cuda.set_device(self.photoai_device)
            event_id = str(time.time_ns())

            # Model switching if needed
            if model_select != self.model.current_model:
                if model_select == 'v0-Q':
                    logger.info('Loading v0-Q checkpoint')
                    self.model.load_state_dict(self.ckpt_Q, strict=False)
                elif model_select == 'v0-F':
                    logger.info('Loading v0-F checkpoint')
                    self.model.load_state_dict(self.ckpt_F, strict=False)
                self.model.current_model = model_select

            # Set dtypes
            self.model.ae_dtype = convert_dtype(ae_dtype)
            self.model.model.dtype = convert_dtype(diff_dtype)

            # Process input image
            if hasattr(input_image, 'convert'):  # PIL Image
                input_image = input_image.convert('RGB')
                LQ = np.array(input_image).astype(np.uint8)
                logger.debug(f"Input PIL image converted to numpy: {LQ.shape}")
            else:  # Already numpy array
                LQ = input_image.astype(np.uint8)
                logger.debug(f"Input already numpy array: {LQ.shape}")

            LQ = HWC3(LQ)
            logger.debug(f"After HWC3: {LQ.shape}")

            LQ = upscale_image(LQ, upscale, unit_resolution=32, min_size=1024)
            logger.debug(f"After upscale by {upscale}x: {LQ.shape}")

            LQ = np.array(LQ) / 255 * 2 - 1
            LQ = torch.tensor(LQ, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.photoai_device)[:, :3, :,
                 :]
            logger.debug(f"Input tensor shape: {LQ.shape}, range: [{LQ.min():.3f}, {LQ.max():.3f}]")

            captions = [prompt]
            logger.debug(f"Using captions: {captions}")
            logger.debug(
                f"Generation parameters - steps: {edm_steps}, cfg: {s_cfg}, stage1: {s_stage1}, stage2: {s_stage2}")

            # Run diffusion sampling
            logger.info("Starting diffusion sampling...")
            samples = self.model.batchify_sample(
                LQ, captions, num_steps=edm_steps, restoration_scale=s_stage1,
                s_churn=s_churn, s_noise=s_noise, cfg_scale=s_cfg, control_scale=s_stage2,
                seed=seed, num_samples=num_samples, p_p=a_prompt, n_p=n_prompt,
                color_fix_type=color_fix_type, use_linear_CFG=linear_CFG,
                use_linear_control_scale=linear_s_stage2, cfg_scale_start=spt_linear_CFG,
                control_scale_start=spt_linear_s_stage2
            )
            logger.debug(f"Diffusion output shape: {samples.shape if hasattr(samples, 'shape') else type(samples)}")
            logger.debug(f"Diffusion output range: [{samples.min():.3f}, {samples.max():.3f}]" if hasattr(samples,
                                                                                                          'min') else "No min/max available")

            # Convert results to PIL Images for Gradio Gallery
            results = []
            for sample in samples:
                sample_np = (sample.permute(1, 2, 0) * 127.5 + 127.5).cpu().numpy().round().clip(0, 255).astype(
                    np.uint8)
                # Convert numpy array to PIL Image for proper display
                sample_pil = Image.fromarray(sample_np)
                results.append(sample_pil)

            logger.info(f"Stage2 processing completed. Generated {len(results)} samples")
            logger.debug(f"Output sample shapes: {[np.array(img).shape for img in results]}")

            # Log some processing info for debugging
            if len(results) > 0:
                first_result = np.array(results[0])
                logger.info(
                    f"First result - shape: {first_result.shape}, dtype: {first_result.dtype}, range: [{first_result.min()}, {first_result.max()}]")

            return results, event_id, 3 if model_select == 'v0-Q' else 1

        except Exception as e:
            logger.error(f"Error in stage2_process: {e}")
            logger.error(traceback.format_exc())
            raise

    def load_and_reset(self, param_setting):
        """Load parameter settings"""
        try:
            # Use actual default_setting values where available
            edm_steps = self.default_setting.edm_steps if hasattr(self.default_setting, 'edm_steps') else 50
            s_stage2 = 1.0
            s_stage1 = -1.0
            s_churn = 0
            s_noise = 1.003
            a_prompt = 'Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.'
            n_prompt = 'painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth'
            gamma_correction = 2.0
            linear_CFG = True
            linear_s_stage2 = False
            spt_linear_s_stage2 = 0.0

            if param_setting == "Quality":
                s_cfg = self.default_setting.s_cfg_Quality if hasattr(self.default_setting, 's_cfg_Quality') else 7.5
                spt_linear_CFG = self.default_setting.spt_linear_CFG_Quality if hasattr(self.default_setting,
                                                                                        'spt_linear_CFG_Quality') else 4.0
            elif param_setting == "Fidelity":
                s_cfg = self.default_setting.s_cfg_Fidelity if hasattr(self.default_setting, 's_cfg_Fidelity') else 4.0
                spt_linear_CFG = self.default_setting.spt_linear_CFG_Fidelity if hasattr(self.default_setting,
                                                                                         'spt_linear_CFG_Fidelity') else 1.0
            else:
                s_cfg = 7.5
                spt_linear_CFG = 4.0

            return (edm_steps, spt_linear_CFG, spt_linear_s_stage2, s_cfg, s_stage2,
                    s_stage1, s_churn, s_noise, a_prompt, n_prompt, gamma_correction,
                    linear_CFG, linear_s_stage2)

        except Exception as e:
            logger.error(f"Error in load_and_reset: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for display"""
        return {
            "photoai_device": self.photoai_device,
            "llava_device": self.llava_device,
            "llava_model": LLAVA_MODEL_PATH if self.use_llava else 'Disabled',
            "llava_status": '✅ Loaded' if self.use_llava else '❌ Disabled',
            "model_type": "PhotoAI",
            "current_model": self.model.current_model if self.model else "Unknown"
        }

    def setup_gradio_interface(self):
        """Setup modern Gradio interface using imageai/update structure"""

        def create_interface():
            with gr.Blocks(title="PhotoAI Demo", theme=gr.themes.Soft()) as interface:
                gr.Markdown("# PhotoAI Demo with Modern UI")
                gr.Markdown("**Modern UI structure from imageai/update + Working PhotoAI functions**")

                with gr.Tab("🎨 Image Enhancement"):
                    with gr.Row():
                        with gr.Column():
                            input_image = gr.Image(label="Input Image", type="pil")
                            gamma_correction = gr.Slider(0.5, 2.0, value=1.0, label="Gamma Correction")
                            stage1_btn = gr.Button("✨ Enhance Image", variant="primary")

                        with gr.Column():
                            stage1_output = gr.Image(label="Enhanced Image", type="numpy")

                    stage1_btn.click(
                        fn=self.stage1_process,
                        inputs=[input_image, gamma_correction],
                        outputs=stage1_output
                    )

                with gr.Tab("🤖 LLaVA Captioning"):
                    with gr.Row():
                        with gr.Column():
                            llava_input = gr.Image(label="Image to Caption", type="pil")

                            with gr.Accordion("LLaVA Settings", open=True):
                                temperature = gr.Slider(0.1, 1.0, value=0.2, label="Temperature")
                                top_p = gr.Slider(0.1, 1.0, value=0.9, label="Top P")
                                custom_prompt = gr.Textbox(
                                    label="Custom Prompt",
                                    value="Describe this image in detail.",
                                    lines=2
                                )

                            llava_btn = gr.Button("🔍 Generate Caption", variant="primary")

                        with gr.Column():
                            caption_output = gr.Textbox(label="Generated Caption", lines=5)

                    llava_btn.click(
                        fn=self.llava_process,
                        inputs=[llava_input, temperature, top_p, custom_prompt],
                        outputs=caption_output
                    )

                with gr.Tab("🚀 Full Pipeline"):
                    with gr.Row():
                        with gr.Column():
                            # Input controls
                            pipeline_input = gr.Image(label="Input Image", type="pil")
                            pipeline_prompt = gr.Textbox(label="Generation Prompt", lines=2)

                            with gr.Accordion("Generation Settings", open=True):
                                with gr.Row():
                                    upscale = gr.Slider(1, 8, value=1, step=1, label="Upscale Factor")
                                    num_samples = gr.Slider(1, 4, value=1, step=1, label="Number of Samples")

                                with gr.Row():
                                    edm_steps = gr.Slider(20, 200, value=50, step=1, label="Steps")
                                    s_cfg = gr.Slider(1.0, 15.0, value=7.5, step=0.1, label="CFG Scale")

                                with gr.Row():
                                    s_stage1 = gr.Slider(-1.0, 6.0, value=-1.0, step=1.0, label="Stage1 Guidance")
                                    s_stage2 = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Stage2 Guidance")

                                seed = gr.Slider(-1, 999999999, value=1234, step=1, label="Seed")

                            with gr.Accordion("Advanced Settings", open=False):
                                with gr.Row():
                                    s_churn = gr.Slider(0, 40, value=0, step=1, label="S-Churn")
                                    s_noise = gr.Slider(1.0, 1.1, value=1.003, step=0.001, label="S-Noise")

                                with gr.Row():
                                    diff_dtype = gr.Radio(['fp32', 'fp16', 'bf16'], label="Diffusion Data Type",
                                                          value="fp16")
                                    ae_dtype = gr.Radio(['fp32', 'bf16'], label="Auto-Encoder Data Type", value="bf16")

                                with gr.Row():
                                    color_fix_type = gr.Radio(["None", "AdaIn", "Wavelet"], label="Color-Fix Type",
                                                              value="Wavelet")
                                    model_select = gr.Radio(["v0-Q", "v0-F"], label="Model Selection", value="v0-Q")

                                with gr.Row():
                                    linear_CFG = gr.Checkbox(label="Linear CFG", value=True)
                                    linear_s_stage2 = gr.Checkbox(label="Linear Stage2 Guidance", value=False)

                                with gr.Row():
                                    spt_linear_CFG = gr.Slider(1.0, 9.0, value=4.0, step=0.5, label="CFG Start")
                                    spt_linear_s_stage2 = gr.Slider(0.0, 1.0, value=0.0, step=0.05,
                                                                    label="Guidance Start")

                                a_prompt = gr.Textbox(
                                    label="Positive Prompt",
                                    value='Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.',
                                    lines=3
                                )
                                n_prompt = gr.Textbox(
                                    label="Negative Prompt",
                                    value='painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth',
                                    lines=3
                                )

                            # Control buttons
                            with gr.Row():
                                param_setting = gr.Dropdown(["Quality", "Fidelity"], label="Param Setting",
                                                            value="Quality")
                                reset_btn = gr.Button("🔄 Reset Parameters")

                            pipeline_btn = gr.Button("🚀 Generate", variant="primary", size="lg")

                        with gr.Column():
                            pipeline_output = gr.Gallery(label="Generated Images", show_label=False)
                            event_output = gr.Textbox(label="Event ID", visible=False)
                            score_output = gr.Number(label="Score", visible=False)

                    # Connect pipeline processing
                    pipeline_btn.click(
                        fn=self.stage2_process,
                        inputs=[
                            pipeline_input, pipeline_prompt, a_prompt, n_prompt, num_samples, upscale,
                            edm_steps, s_stage1, s_stage2, s_cfg, seed, s_churn, s_noise,
                            color_fix_type, diff_dtype, ae_dtype, gamma_correction,
                            linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2, model_select
                        ],
                        outputs=[pipeline_output, event_output, score_output]
                    )

                    # Connect reset button
                    reset_btn.click(
                        fn=self.load_and_reset,
                        inputs=[param_setting],
                        outputs=[
                            edm_steps, spt_linear_CFG, spt_linear_s_stage2, s_cfg, s_stage2,
                            s_stage1, s_churn, s_noise, a_prompt, n_prompt, gamma_correction,
                            linear_CFG, linear_s_stage2
                        ]
                    )

                # Model Information
                with gr.Accordion("ℹ️ Model Information", open=False):
                    model_info = self.get_model_info()
                    gr.Markdown(f"""
                    **PhotoAI Device:** {model_info['photoai_device']}  
                    **LLaVA Device:** {model_info['llava_device']}  
                    **LLaVA Model:** {model_info['llava_model']}  
                    **LLaVA Status:** {model_info['llava_status']}  
                    **Model Type:** {model_info['model_type']}  
                    **Current Model:** {model_info['current_model']}
                    """)

            return interface

        self.interface = create_interface()

    def launch(self):
        """Launch the Gradio interface"""
        self.setup_gradio_interface()
        self.interface.launch(
            server_name=self.args.ip,
            server_port=self.args.port,
            share=True,
            show_error=True
        )


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser()

    # Server options
    parser.add_argument("--ip", type=str, default='127.0.0.1')
    parser.add_argument("--port", type=int, default=6688)
    parser.add_argument("--opt", type=str, default='options/PhotoAI_v0.yaml')

    # Model options
    parser.add_argument("--no_llava", action='store_true', default=False)
    parser.add_argument("--loading_half_params", action='store_true', default=False)
    parser.add_argument("--use_tile_vae", action='store_true', default=False)
    parser.add_argument("--encoder_tile_size", type=int, default=512)
    parser.add_argument("--decoder_tile_size", type=int, default=64)
    parser.add_argument("--load_8bit_llava", action='store_true', default=False)
    parser.add_argument("--debug", action='store_true', default=True)

    return parser


def main():
    """Main function with debugging"""
    logger.info("=" * 60)
    logger.info("STARTING HYBRID PHOTOAI DEMO")
    logger.info("=" * 60)

    try:
        parser = create_parser()
        args = parser.parse_args()

        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("Debug mode enabled")

        logger.info(f"Arguments: {vars(args)}")

        # Create and launch demo
        demo = HybridImageAIDemo(args)
        demo.launch()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        logger.error("FATAL ERROR:")
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()