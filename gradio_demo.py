import argparse
import copy
import logging
import os
import sys
import time
import traceback

import gradio as gr
import numpy as np
import torch
from PIL import Image
from gradio_imageslider import ImageSlider

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from CKPT_PTH import LLAVA_MODEL_PATH
from photoai.util import HWC3, upscale_image, fix_resize, convert_dtype, create_photoai_model
from photoai.util import load_QF_ckpt

# Setup logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--opt", type=str, default='options/PhotoAI_v0.yaml')
parser.add_argument("--ip", type=str, default='127.0.0.1')
parser.add_argument("--port", type=int, default='6688')
parser.add_argument("--no_llava", action='store_true', default=True)
parser.add_argument("--use_image_slider", action='store_true', default=False)
parser.add_argument("--log_history", action='store_true', default=False)
parser.add_argument("--loading_half_params", action='store_true', default=False)
parser.add_argument("--use_tile_vae", action='store_true', default=False)
parser.add_argument("--encoder_tile_size", type=int, default=512)
parser.add_argument("--decoder_tile_size", type=int, default=64)
parser.add_argument("--load_8bit_llava", action='store_true', default=False)
parser.add_argument("--debug", action='store_true', default=False, help="Enable debug mode with verbose logging")
args = parser.parse_args()

# Set debug logging level if enabled
if args.debug:
    logging.getLogger().setLevel(logging.DEBUG)
    logger.info("Debug mode enabled")

logger.info(f"Starting with args: {args}")

server_ip = args.ip
server_port = args.port
use_llava = not args.no_llava

logger.info(f"LLaVA enabled: {use_llava}")
logger.info(f"CUDA devices available: {torch.cuda.device_count()}")

if torch.cuda.device_count() >= 2:
    photo_ai_device = 'cuda:0'
    llava_device = 'cuda:1'
elif torch.cuda.device_count() == 1:
    photo_ai_device = 'cuda:0'
    llava_device = 'cuda:0'
else:
    raise ValueError('Currently support CUDA only.')

logger.info(f"PhotoAI device: {photo_ai_device}, LLaVA device: {llava_device}")

try:
    logger.info("Loading PhotoAI model...")
    model, default_setting = create_photoai_model(args.opt, photoai_sign='Q', load_default_setting=True)
    if args.loading_half_params:
        logger.info("Converting model to half precision")
        model = model.half()
    if args.use_tile_vae:
        logger.info(
            f"Initializing tile VAE with sizes: encoder={args.encoder_tile_size}, decoder={args.decoder_tile_size}")
        model.init_tile_vae(encoder_tile_size=args.encoder_tile_size, decoder_tile_size=args.decoder_tile_size)
    model = model.to(photo_ai_device)
    model.first_stage_model.denoise_encoder_s1 = copy.deepcopy(model.first_stage_model.denoise_encoder)
    model.current_model = 'v0-Q'
    ckpt_Q, ckpt_F = load_QF_ckpt(args.opt)
    logger.info("PhotoAI model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load PhotoAI model: {e}")
    logger.error(traceback.format_exc())
    raise

# load LLaVA using Transformers
if use_llava:
    try:
        logger.info("Loading LLaVA model using Transformers...")
        logger.info(f"LLaVA model path: {LLAVA_MODEL_PATH}")
        logger.info(f"8-bit loading: {args.load_8bit_llava}")

        llava_processor = LlavaNextProcessor.from_pretrained(LLAVA_MODEL_PATH)
        logger.info("LLaVA processor loaded successfully")

        if args.load_8bit_llava:
            logger.info("Loading LLaVA model in 8-bit mode...")
            llava_model = LlavaNextForConditionalGeneration.from_pretrained(
                LLAVA_MODEL_PATH,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map=llava_device
            )
        else:
            logger.info("Loading LLaVA model in 16-bit mode...")
            llava_model = LlavaNextForConditionalGeneration.from_pretrained(
                LLAVA_MODEL_PATH,
                torch_dtype=torch.float16,
                device_map=llava_device
            )
        llava_model.eval()
        logger.info("LLaVA model loaded and set to eval mode successfully!")

    except Exception as e:
        logger.error(f"Failed to load LLaVA model: {e}")
        logger.error(traceback.format_exc())
        logger.warning("Continuing without LLaVA - setting use_llava to False")
        use_llava = False
        llava_processor = None
        llava_model = None
else:
    logger.info("LLaVA disabled by user")
    llava_processor = None
    llava_model = None


def generate_llava_caption(image, temperature=0.2, top_p=0.9, qs=None):
    """Generate image caption using Transformers LLaVA"""
    logger.debug(f"generate_llava_caption called with temperature={temperature}, top_p={top_p}")

    if not use_llava or llava_model is None:
        logger.warning("LLaVA not available for caption generation")
        return "LLaVA is not available. Please add text manually."

    try:
        # Default question for image captioning
        if qs is None:
            qs = "Describe this image in detail."

        logger.debug(f"Using question: {qs}")
        logger.debug(f"Input image size: {image.size if hasattr(image, 'size') else 'Unknown'}")

        # Prepare conversation format expected by LLaVA
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
        logger.debug("Applying chat template...")
        prompt = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
        logger.debug(f"Generated prompt length: {len(prompt)}")

        logger.debug("Processing inputs...")
        inputs = llava_processor(text=prompt, images=image, return_tensors="pt").to(llava_device)
        logger.debug(
            f"Input tensor shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in inputs.items()]}")

        # Generate caption with specified parameters
        logger.debug("Starting generation...")
        start_time = time.time()
        with torch.no_grad():
            output = llava_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=llava_processor.tokenizer.eos_token_id
            )
        generation_time = time.time() - start_time
        logger.debug(f"Generation completed in {generation_time:.2f} seconds")

        # Decode and extract caption
        logger.debug("Decoding output...")
        generated_text = llava_processor.decode(output[0], skip_special_tokens=True)
        logger.debug(f"Raw generated text: {generated_text}")

        # Extract just the assistant's response
        if "assistant\n" in generated_text:
            caption = generated_text.split("assistant\n")[-1].strip()
        else:
            caption = generated_text.strip()

        logger.info(f"Generated caption: {caption}")
        return caption

    except Exception as e:
        logger.error(f"Error generating LLaVA caption: {e}")
        logger.error(traceback.format_exc())
        return "Error generating caption. Please add text manually."


def stage1_process(input_image, gamma_correction):
    logger.info("Starting Stage1 processing...")
    logger.debug(f"Gamma correction: {gamma_correction}")

    try:
        torch.cuda.set_device(photo_ai_device)
        LQ = HWC3(input_image)
        logger.debug(f"Image after HWC3: {LQ.shape}")

        LQ = fix_resize(LQ, 512)
        logger.debug(f"Image after resize: {LQ.shape}")

        # stage1
        LQ = np.array(LQ) / 255 * 2 - 1
        LQ = torch.tensor(LQ, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(photo_ai_device)[:, :3, :, :]
        logger.debug(f"Tensor shape before denoise: {LQ.shape}")

        LQ = model.batchify_denoise(LQ, is_stage1=True)
        logger.debug(f"Tensor shape after denoise: {LQ.shape}")

        LQ = (LQ[0].permute(1, 2, 0) * 127.5 + 127.5).cpu().numpy().round().clip(0, 255).astype(np.uint8)
        logger.debug(f"Final array shape: {LQ.shape}")

        # gamma correction
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


def llava_process(input_image, temperature, top_p, qs=None):
    logger.info("Starting LLaVA processing...")
    logger.debug(f"Parameters: temperature={temperature}, top_p={top_p}")

    try:
        torch.cuda.set_device(llava_device)
        if use_llava:
            LQ = HWC3(input_image)
            logger.debug(f"Image processed with HWC3: {LQ.shape}")

            LQ = Image.fromarray(LQ.astype('uint8'))
            logger.debug(f"Converted to PIL Image: {LQ.size}")

            caption = generate_llava_caption(LQ, temperature=temperature, top_p=top_p, qs=qs)
            logger.info(f"LLaVA processing completed: {caption}")
        else:
            caption = 'LLaVA is not available. Please add text manually.'
            logger.warning("LLaVA not available, returning default message")

        return caption

    except Exception as e:
        logger.error(f"Error in llave_process: {e}")
        logger.error(traceback.format_exc())
        return f"Error in LLaVA processing: {str(e)}"


def stage2_process(input_image, prompt, a_prompt, n_prompt, num_samples, upscale, edm_steps, s_stage1, s_stage2,
                   s_cfg, seed, s_churn, s_noise, color_fix_type, diff_dtype, ae_dtype, gamma_correction,
                   linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2, model_select):
    logger.info("Starting Stage2 processing...")
    logger.debug(f"Parameters: prompt='{prompt}', upscale={upscale}, steps={edm_steps}, model={model_select}")

    try:
        torch.cuda.set_device(photo_ai_device)
        event_id = str(time.time_ns())
        event_dict = {'event_id': event_id, 'localtime': time.ctime(), 'prompt': prompt, 'a_prompt': a_prompt,
                      'n_prompt': n_prompt, 'num_samples': num_samples, 'upscale': upscale, 'edm_steps': edm_steps,
                      's_stage1': s_stage1, 's_stage2': s_stage2, 's_cfg': s_cfg, 'seed': seed, 's_churn': s_churn,
                      's_noise': s_noise, 'color_fix_type': color_fix_type, 'diff_dtype': diff_dtype,
                      'ae_dtype': ae_dtype,
                      'gamma_correction': gamma_correction, 'linear_CFG': linear_CFG,
                      'linear_s_stage2': linear_s_stage2,
                      'spt_linear_CFG': spt_linear_CFG, 'spt_linear_s_stage2': spt_linear_s_stage2,
                      'model_select': model_select}

        logger.debug(f"Event ID: {event_id}")

        if model_select != model.current_model:
            logger.info(f"Switching model from {model.current_model} to {model_select}")
            if model_select == 'v0-Q':
                logger.info('Loading v0-Q checkpoint')
                model.load_state_dict(ckpt_Q, strict=False)
            elif model_select == 'v0-F':
                logger.info('Loading v0-F checkpoint')
                model.load_state_dict(ckpt_F, strict=False)
            model.current_model = model_select

        model.ae_dtype = convert_dtype(ae_dtype)
        model.model.dtype = convert_dtype(diff_dtype)
        logger.debug(f"Set dtypes: ae={ae_dtype}, diff={diff_dtype}")

        LQ = HWC3(input_image)
        logger.debug(f"Input image shape: {LQ.shape}")

        LQ = upscale_image(LQ, upscale, unit_resolution=32, min_size=1024)
        logger.debug(f"After upscale: {LQ.shape}")

        LQ = np.array(LQ) / 255 * 2 - 1
        LQ = torch.tensor(LQ, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(photo_ai_device)[:, :3, :, :]
        logger.debug(f"Tensor shape: {LQ.shape}")

        captions = [prompt]
        logger.debug(f"Using captions: {captions}")

        logger.info("Starting diffusion sampling...")
        start_time = time.time()
        samples = model.batchify_sample(LQ, captions, num_steps=edm_steps, restoration_scale=s_stage1, s_churn=s_churn,
                                        s_noise=s_noise, cfg_scale=s_cfg, control_scale=s_stage2, seed=seed,
                                        num_samples=num_samples, p_p=a_prompt, n_p=n_prompt,
                                        color_fix_type=color_fix_type,
                                        use_linear_CFG=linear_CFG, use_linear_control_scale=linear_s_stage2,
                                        cfg_scale_start=spt_linear_CFG, control_scale_start=spt_linear_s_stage2)
        sampling_time = time.time() - start_time
        logger.info(f"Diffusion sampling completed in {sampling_time:.2f} seconds")

        results = []
        for i, sample in enumerate(samples):
            sample = (sample.permute(1, 2, 0) * 127.5 + 127.5).cpu().numpy().round().clip(0, 255).astype(np.uint8)
            results.append(sample)
            logger.debug(f"Processed sample {i}: {sample.shape}")

        if args.log_history:
            logger.info(f"Saving history for event {event_id[:5]}")
            os.makedirs(f'./history/{event_id[:5]}', exist_ok=True)
            with open(f'./history/{event_id[:5]}/logs', 'a') as f:
                f.write(str(event_dict) + '\n')
            f.close()
            input_image.save(f'./history/{event_id[:5]}/LQ.png')
            for i, result in enumerate(results):
                Image.fromarray(result).save(f'./history/{event_id[:5]}/HQ_{i}.png')

        logger.info(f"Stage2 processing completed successfully. Generated {len(results)} samples")
        return results, event_id, 3 if model_select == 'v0-Q' else 1

    except Exception as e:
        logger.error(f"Error in stage2_process: {e}")
        logger.error(traceback.format_exc())
        raise_start = spt_linear_s_stage2

        results = []
        for sample in samples:
            sample = (sample.permute(1, 2, 0) * 127.5 + 127.5).cpu().numpy().round().clip(0, 255).astype(np.uint8)
        results.append(sample)

        if args.log_history:
            os.makedirs(f'./history/{event_id[:5]}', exist_ok=True)
        with open(f'./history/{event_id[:5]}/logs', 'a') as f:
            f.write(str(event_dict) + '\n')
        f.close()
        input_image.save(f'./history/{event_id[:5]}/LQ.png')
        for i, result in enumerate(results):
            Image.fromarray(result).save(f'./history/{event_id[:5]}/HQ_{i}.png')

    return results, event_id, 3 if model_select == 'v0-Q' else 1


def load_and_reset(param_setting):
    logger.info(f"Loading parameter setting: {param_setting}")

    try:
        if param_setting == "Quality":
            result = (default_setting.edm_steps, default_setting.spt_linear_CFG_Quality,
                      default_setting.spt_linear_s_stage2_Quality, default_setting.s_cfg_Quality,
                      default_setting.s_stage2_Quality, default_setting.s_stage1_Quality,
                      default_setting.s_churn, default_setting.s_noise, default_setting.a_prompt,
                      default_setting.n_prompt, 2.0, default_setting.linear_CFG, default_setting.linear_s_stage2)
        elif param_setting == "Fidelity":
            result = (default_setting.edm_steps, default_setting.spt_linear_CFG_Fidelity,
                      default_setting.spt_linear_s_stage2_Fidelity, default_setting.s_cfg_Fidelity,
                      default_setting.s_stage2_Fidelity, default_setting.s_stage1_Fidelity,
                      default_setting.s_churn, default_setting.s_noise, default_setting.a_prompt,
                      default_setting.n_prompt, 2.0, default_setting.linear_CFG, default_setting.linear_s_stage2)
        else:
            logger.warning(f"Unknown parameter setting: {param_setting}, using Quality as default")
            result = (default_setting.edm_steps, default_setting.spt_linear_CFG_Quality,
                      default_setting.spt_linear_s_stage2_Quality, default_setting.s_cfg_Quality,
                      default_setting.s_stage2_Quality, default_setting.s_stage1_Quality,
                      default_setting.s_churn, default_setting.s_noise, default_setting.a_prompt,
                      default_setting.n_prompt, 2.0, default_setting.linear_CFG, default_setting.linear_s_stage2)

        logger.debug(f"Loaded parameters: {result}")
        return result

    except Exception as e:
        logger.error(f"Error in load_and_reset: {e}")
        logger.error(traceback.format_exc())
        raise


title_md = """
# **Erik's Photos: Scaling Up to Excellence** 🔧 DEBUG MODE

🐛 **Debug Features Enabled:**
- Comprehensive logging for all operations
- Detailed error tracking and stack traces  
- Performance timing measurements
- Model loading and parameter monitoring
- Memory usage tracking
"""


# Debug function to capture logs
def update_debug_info():
    return get_debug_info()


# Enhanced button functions with debug output
def debug_stage1_process(input_image, gamma_correction):
    try:
        result = stage1_process(input_image, gamma_correction)
        return result, "✅ Stage1 completed successfully"
    except Exception as e:
        error_msg = f"❌ Stage1 failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return None, error_msg


def debug_llava_process(input_image, temperature, top_p):
    try:
        result = llava_process(input_image, temperature, top_p)
        return result, f"✅ LLaVA completed: Generated caption with {len(result)} characters"
    except Exception as e:
        error_msg = f"❌ LLaVA failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return "Error in LLaVA processing", error_msg


def debug_stage2_process(input_image, prompt, a_prompt, n_prompt, num_samples, upscale, edm_steps, s_stage1,
                         s_stage2,
                         s_cfg, seed, s_churn, s_noise, color_fix_type, diff_dtype, ae_dtype, gamma_correction,
                         linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2, model_select):
    try:
        start_time = time.time()
        results, event_id, score = stage2_process(input_image, prompt, a_prompt, n_prompt, num_samples, upscale,
                                                  edm_steps, s_stage1, s_stage2, s_cfg, seed, s_churn, s_noise,
                                                  color_fix_type, diff_dtype, ae_dtype, gamma_correction,
                                                  linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2,
                                                  model_select)
        total_time = time.time() - start_time
        debug_msg = f"✅ Stage2 completed in {total_time:.2f}s\nEvent ID: {event_id}\nGenerated {len(results)} samples"
        return results, event_id, score, debug_msg
    except Exception as e:
        error_msg = f"❌ Stage2 failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return [], "", 0, error_msg


# Create a debug info function
def get_debug_info():
    info = []
    info.append(f"🔧 **System Info:**")
    info.append(f"- CUDA Available: {torch.cuda.is_available()}")
    info.append(f"- CUDA Device Count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        info.append(f"- Current CUDA Device: {torch.cuda.current_device()}")
        info.append(f"- CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    info.append(f"- PhotoAI Device: {photo_ai_device}")
    info.append(f"- LLaVA Device: {llava_device}")
    info.append(f"- LLaVA Enabled: {use_llava}")
    info.append(f"- Model Path: {LLAVA_MODEL_PATH}")
    info.append(f"- Debug Mode: {args.debug}")
    return "\n".join(info)


logger.info("Creating Gradio interface...")

with gr.Blocks(title="Erik''s PhotoAI") as interface:
    gr.Markdown(title_md)

    # Debug info section
    with gr.Row():
        with gr.Column():
            debug_info = gr.Markdown(get_debug_info())
            refresh_debug = gr.Button("🔄 Refresh Debug Info")

    with gr.Row():
        with gr.Column():
            gr.Markdown("<center>Input</center>")
            input_image = gr.Image(label="Upload an image", type="pil")
            upscale = gr.Slider(label="Upscale Factor", minimum=1, maximum=8, value=1, step=1)
            with gr.Group():
                prompt = gr.Textbox(label="Text Prompt", placeholder="Describe the image", value="")
                with gr.Row():
                    llava_temperature = gr.Slider(label="LLaVA Temperature", minimum=0.1, maximum=1.0, value=0.2,
                                                  step=0.1)
                    llava_top_p = gr.Slider(label="LLaVA Top P", minimum=0.1, maximum=1.0, value=0.9, step=0.1)

            with gr.Group():
                gamma_correction = gr.Slider(label="Gamma Correction", minimum=0.1, maximum=2.0, value=1.0, step=0.1)
                edm_steps = gr.Slider(label="Steps", minimum=20, maximum=200, value=50, step=1)

                with gr.Row():
                    s_cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=15.0, value=7.5, step=0.1)
                    s_stage2 = gr.Slider(label="Stage2 Guidance", minimum=0., maximum=1., value=1.0, step=0.05)
                    s_stage1 = gr.Slider(label="Stage1 Guidance", minimum=-1.0, maximum=6.0, value=-1.0, step=1.0)
                    seed = gr.Slider(label="Seed", minimum=-1, maximum=999999999, value=1234, step=1)
                    s_churn = gr.Slider(label="S-Churn", minimum=0, maximum=40, value=0, step=1)
                    s_noise = gr.Slider(label="S-Noise", minimum=1.0, maximum=1.1, value=1.003, step=0.001)
                    num_samples = gr.Slider(label="Num Samples", minimum=1,
                                            maximum=4 if not args.use_image_slider else 1
                                            , value=1, step=1)

                a_prompt = gr.Textbox(label="Default Positive Prompt",
                                      value='Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.')
                n_prompt = gr.Textbox(label="Default Negative Prompt",
                                      value='painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth')
                with gr.Row():
                    with gr.Column():
                        linear_CFG = gr.Checkbox(label="Linear CFG", value=True)
                        spt_linear_CFG = gr.Slider(label="CFG Start", minimum=1.0,
                                                   maximum=9.0, value=4.0, step=0.5)
                    with gr.Column():
                        linear_s_stage2 = gr.Checkbox(label="Linear Stage2 Guidance", value=False)
                        spt_linear_s_stage2 = gr.Slider(label="Guidance Start", minimum=0.,
                                                        maximum=1., value=0.0, step=0.05)
                with gr.Row():
                    with gr.Column():
                        diff_dtype = gr.Radio(['fp32', 'fp16', 'bf16'], label="Diffusion Data Type", value="fp16",
                                              interactive=True)
                    with gr.Column():
                        ae_dtype = gr.Radio(['fp32', 'bf16'], label="Auto-Encoder Data Type", value="bf16",
                                            interactive=True)
                    with gr.Column():
                        color_fix_type = gr.Radio(["None", "AdaIn", "Wavelet"], label="Color-Fix Type", value="Wavelet",
                                                  interactive=True)
                    with gr.Column():
                        model_select = gr.Radio(["v0-Q", "v0-F"], label="Model Selection", value="v0-Q",
                                                interactive=True)

        with gr.Column():
            gr.Markdown("<center>Stage2 Output</center>")
            if not args.use_image_slider:
                result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery1")
            else:
                result_gallery = ImageSlider(label='Output', show_label=False, elem_id="gallery1")
            with gr.Row():
                with gr.Column():
                    denoise_button = gr.Button(value="Stage1 Run")
                with gr.Column():
                    llave_button = gr.Button(value="LlaVa Run")
                with gr.Column():
                    diffusion_button = gr.Button(value="Stage2 Run")
            with gr.Row():
                with gr.Column():
                    param_setting = gr.Dropdown(["Quality", "Fidelity"], interactive=True, label="Param Setting",
                                                value="Quality")
                with gr.Column():
                    restart_button = gr.Button(value="Reset Param", scale=2)
            with gr.Accordion("Feedback & Debug", open=True):
                fb_score = gr.Slider(label="Feedback Score", minimum=1, maximum=5, value=3, step=1,
                                     interactive=True)
                fb_text = gr.Textbox(label="Feedback Text", value="", placeholder='Please enter your feedback here.')
                fb_button = gr.Button(value="Submit Feedback")

                # Debug output area
                debug_output = gr.Textbox(label="Debug Output", lines=10, value="Debug logs will appear here...",
                                          interactive=False)


    # Debug function to capture logs
    def update_debug_info():
        return get_debug_info()


    # Enhanced button functions with debug output
    def debug_stage1_process(input_image, gamma_correction):
        try:
            result = stage1_process(input_image, gamma_correction)
            return result, "✅ Stage1 completed successfully"
        except Exception as e:
            error_msg = f"❌ Stage1 failed: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return None, error_msg


    def debug_llava_process(input_image, temperature, top_p):
        try:
            result = llava_process(input_image, temperature, top_p)
            return result, f"✅ LLaVA completed: Generated caption with {len(result)} characters"
        except Exception as e:
            error_msg = f"❌ LLaVA failed: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return "Error in LLaVA processing", error_msg


    def debug_stage2_process(input_image, prompt, a_prompt, n_prompt, num_samples, upscale, edm_steps, s_stage1,
                             s_stage2,
                             s_cfg, seed, s_churn, s_noise, color_fix_type, diff_dtype, ae_dtype, gamma_correction,
                             linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2, model_select):
        try:
            start_time = time.time()
            results, event_id, score = stage2_process(input_image, prompt, a_prompt, n_prompt, num_samples, upscale,
                                                      edm_steps, s_stage1, s_stage2, s_cfg, seed, s_churn, s_noise,
                                                      color_fix_type, diff_dtype, ae_dtype, gamma_correction,
                                                      linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2,
                                                      model_select)
            total_time = time.time() - start_time
            debug_msg = f"✅ Stage2 completed in {total_time:.2f}s\nEvent ID: {event_id}\nGenerated {len(results)} samples"
            return results, event_id, score, debug_msg
        except Exception as e:
            error_msg = f"❌ Stage2 failed: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return [], "", 0, error_msg


    # Connect the enhanced functions
    refresh_debug.click(fn=update_debug_info, outputs=debug_info)

    denoise_button.click(fn=debug_stage1_process,
                         inputs=[input_image, gamma_correction],
                         outputs=[input_image, debug_output])

    llave_button.click(fn=debug_llava_process,
                       inputs=[input_image, llava_temperature, llava_top_p],
                       outputs=[prompt, debug_output])

    diffusion_button.click(fn=debug_stage2_process,
                           inputs=[input_image, prompt, a_prompt, n_prompt, num_samples, upscale,
                                   edm_steps, s_stage1, s_stage2, s_cfg, seed, s_churn, s_noise,
                                   color_fix_type, diff_dtype, ae_dtype, gamma_correction,
                                   linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2,
                                   model_select],
                           outputs=[result_gallery, fb_text, fb_score, debug_output])

    restart_button.click(fn=load_and_reset, inputs=[param_setting],
                         outputs=[edm_steps, spt_linear_CFG, spt_linear_s_stage2,
                                  s_cfg, s_stage2, s_stage1, s_churn, s_noise,
                                  a_prompt, n_prompt, gamma_correction, linear_CFG,
                                  linear_s_stage2])


def main():
    """Main function with comprehensive debugging and error handling"""
    logger.info("=" * 60)
    logger.info("STARTING SUPIR GRADIO DEMO WITH DEBUG MODE")
    logger.info("=" * 60)

    try:
        # Log startup information
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name} ({props.total_memory / 1024 ** 3:.1f} GB)")

        logger.info(f"Arguments: {vars(args)}")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"LLaVA model path: {LLAVA_MODEL_PATH}")
        logger.info(f"LLaVA path exists: {os.path.exists(LLAVA_MODEL_PATH)}")

        # Test CUDA setup
        if torch.cuda.is_available():
            logger.info("Testing CUDA functionality...")
            test_tensor = torch.randn(10, 10).cuda()
            logger.info(f"CUDA test successful: tensor device = {test_tensor.device}")
            del test_tensor
            torch.cuda.empty_cache()

        # Launch the interface
        logger.info(f"Launching Gradio interface on {server_ip}:{server_port}")
        logger.info(f"Debug mode: {'ENABLED' if args.debug else 'DISABLED'}")
        logger.info(f"LLaVA: {'ENABLED' if use_llava else 'DISABLED'}")
        logger.info(f"Image slider: {'ENABLED' if args.use_image_slider else 'DISABLED'}")
        logger.info(f"History logging: {'ENABLED' if args.log_history else 'DISABLED'}")

        # Check if interface was created successfully
        if 'interface' in globals():
            logger.info("Gradio interface created successfully")
            logger.info("Starting server...")
            interface.queue().launch(
                server_name=server_ip,
                server_port=server_port,
                share=True,
                debug=args.debug,
                show_error=True
            )
        else:
            raise RuntimeError("Failed to create Gradio interface")

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down gracefully...")
        sys.exit(0)

    except Exception as e:
        logger.error("=" * 60)
        logger.error("FATAL ERROR DURING STARTUP")
        logger.error("=" * 60)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        logger.error("=" * 60)

        # Try to provide helpful debugging info
        logger.info("DEBUGGING INFORMATION:")
        logger.info(f"- Current working directory: {os.getcwd()}")
        logger.info(f"- Python executable: {sys.executable}")
        logger.info(f"- Arguments passed: {vars(args)}")

        # Check file paths
        paths_to_check = [
            args.opt,
            LLAVA_MODEL_PATH,
            "./options/",
            "./history/"
        ]

        for path in paths_to_check:
            if os.path.exists(path):
                logger.info(f"- ✅ Path exists: {path}")
            else:
                logger.error(f"- ❌ Path missing: {path}")

        # Memory info
        if torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
                    reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
                    logger.info(f"- GPU {i} memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
            except Exception as mem_e:
                logger.warning(f"Could not get memory info: {mem_e}")

        logger.error("Exiting due to fatal error")
        sys.exit(1)


if __name__ == "__main__":
    main()