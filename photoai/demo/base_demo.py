import sys
import argparse
import json
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    BitsAndBytesConfig
)

from photoai.util import HWC3, upscale_image
from photoai.util import create_photoai_model

# Import and apply OpenCLIP patches - this must be done EARLY
try:
    from photoai.utils.openclip_patch import apply_all_openclip_patches
    print("OpenCLIP patch module imported successfully")
except ImportError as e:
    print(f"Failed to import openclip_patch: {e}")
    print("Make sure openclip_patch.py is in the same directory as this script")
    sys.exit(1)

class ImageAIDemoBase(ABC):
    """Enhanced base class for PhotoAI demos with universal tiled processing support"""

    def __init__(self, args):
        self.llava_model = None
        self.use_llava = None
        self.llava_device = None
        self.imageai_device = None
        self.llava_processor = None
        self.args = args
        self.model: Optional[Any] = None  # PhotoAI model
        self.default_setting = None

        # Tiling configuration - can be overridden by subclasses
        self.tile_size = getattr(args, 'tile_size', 128)  # Latent space tile size
        self.tile_stride = getattr(args, 'tile_stride', 64)  # Latent space stride
        self.enable_tiling = getattr(args, 'enable_tiling', False)  # Default off unless specified

        # Apply OpenCLIP patches BEFORE loading any models
        print("Applying OpenCLIP compatibility patches...")
        if not apply_all_openclip_patches():
            print("Some OpenCLIP patches failed to apply, but continuing...")

        # Initialize components
        self.setup_devices()
        self.setup_imageai_model()
        self.setup_llava_model()

    def setup_devices(self):
        """Setup CUDA devices"""
        if torch.cuda.device_count() >= 2:
            self.imageai_device = 'cuda:0'
            self.llava_device = 'cuda:1'
        elif torch.cuda.device_count() == 1:
            self.imageai_device = 'cuda:0'
            self.llava_device = 'cuda:0'
        else:
            raise ValueError('Currently support CUDA only.')

        print(f"ImageAI device: {self.imageai_device}")
        print(f"LLaVA device: {self.llava_device}")

    def setup_imageai_model(self):
        """Setup ImageAI model - to be implemented by subclasses"""
        pass

    def setup_llava_model(self):
        """Setup LLaVA model with native HuggingFace transformers"""
        self.use_llava = not self.args.no_llava
        self.llava_model = None
        self.llava_processor = None

        if not self.use_llava:
            print("LLaVA disabled by --no_llava flag")
            return

        print(f"Loading LLaVA model: {self.args.llava_model}")

        # Set up quantization if requested
        quantization_config = None
        if self.args.load_8bit_llava:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            print("Using 8-bit quantization for LLaVA")
        elif getattr(self.args, 'load_4bit_llava', False):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            print("Using 4-bit quantization for LLaVA")

        # Load model and processor
        try:
            self.llava_model = LlavaNextForConditionalGeneration.from_pretrained(
                self.args.llava_model,
                torch_dtype=torch.float16,
                device_map=self.llava_device,
                quantization_config=quantization_config
            )
            self.llava_processor = LlavaNextProcessor.from_pretrained(self.args.llava_model)
            print("✅ LLaVA loaded successfully!")

        except Exception as e:
            print(f"⚠️ Failed to load LLaVA: {e}")
            self.use_llava = False

    def llava_caption_process(self, input_image, temperature=0.7, top_p=0.9, custom_prompt=None):
        """Process image for captioning with LLaVA"""
        if not self.use_llava or input_image is None:
            return "LLaVA not available or no image provided."

        try:
            torch.cuda.set_device(self.llava_device)

            LQ = HWC3(input_image)
            pil_image = Image.fromarray(LQ.astype('uint8'))

            prompt = custom_prompt if custom_prompt else "Describe this image in detail."

            # Prepare inputs
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": pil_image}
                    ]
                }
            ]

            prompt_text = self.llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.llava_processor(
                images=pil_image,
                text=prompt_text,
                return_tensors="pt"
            ).to(self.llava_device)

            # Generate
            with torch.no_grad():
                outputs = self.llava_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.llava_processor.tokenizer.eos_token_id
                )

            # Decode response
            response = self.llava_processor.decode(outputs[0], skip_special_tokens=True)
            # Extract generated text after the prompt
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant")[1].strip()

            return response

        except Exception as e:
            error_msg = f"Error in LLaVA captioning: {str(e)}"
            print(error_msg)
            return error_msg

    def stage1_process(self, input_image, gamma_correction=1.0):
        """Stage 1 processing - image enhancement using PhotoAI's methods"""
        if input_image is None:
            return input_image

        try:
            torch.cuda.set_device(self.imageai_device)

            # Prepare input using PhotoAI's expected format
            input_image = HWC3(input_image)
            LQ = np.array(input_image) / 255.0
            LQ = np.power(LQ, gamma_correction)
            LQ *= 255.0
            LQ = LQ.round().clip(0, 255).astype(np.uint8)
            LQ = LQ / 255 * 2 - 1  # PhotoAI expects [-1, 1] range
            LQ = torch.tensor(LQ, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.imageai_device)[:, :3, :,
                 :]

            # Process with PhotoAI model
            with torch.no_grad():
                if hasattr(self.model, 'encode_first_stage_with_denoise'):
                    # Use PhotoAI's built-in denoising method
                    enhanced_latent = self.model.encode_first_stage_with_denoise(LQ, use_sample=False)
                    enhanced_tensor = self.model.decode_first_stage(enhanced_latent)
                else:
                    # Fallback: just encode and decode
                    enhanced_latent = self.model.encode_first_stage(LQ)
                    enhanced_tensor = self.model.decode_first_stage(enhanced_latent)

            # Convert back to numpy in [0, 255] range
            enhanced_image = (enhanced_tensor[0] + 1.0) / 2.0  # Convert from [-1,1] to [0,1]
            enhanced_image = (enhanced_image.permute(1, 2, 0) * 255).cpu().numpy().round().clip(0, 255).astype(np.uint8)
            return enhanced_image

        except Exception as e:
            error_msg = f"Error in stage 1 processing: {str(e)}"
            print(error_msg)
            return input_image

    def _sliding_windows(self, height: int, width: int,
                         tile_size: int, tile_stride: int) -> List[Tuple[int, int, int, int]]:
        """Generate sliding window coordinates for tiling"""
        windows = []
        for hi in range(0, height, tile_stride):
            hi_end = min(hi + tile_size, height)
            for wi in range(0, width, tile_stride):
                wi_end = min(wi + tile_size, width)
                windows.append((hi, hi_end, wi, wi_end))
        return windows

    def analyze_tile_coverage(self, input_image, show_grid=True, tile_size=None, tile_stride=None):
        """Analyze and visualize tile coverage for given image"""
        if input_image is None:
            return input_image, 0

        # Use provided tile sizes or defaults
        if tile_size is None:
            tile_size = self.tile_size
        if tile_stride is None:
            tile_stride = self.tile_stride

        h, w = input_image.shape[:2]

        # Convert latent tile sizes to image space (approximately 8x scaling)
        image_tile_size = tile_size * 8
        image_tile_stride = tile_stride * 8

        tiles_iterator = self._sliding_windows(h, w, image_tile_size, image_tile_stride)
        tile_count = len(list(tiles_iterator))

        if show_grid:
            # Create visualization with grid overlay
            vis_image = input_image.copy()

            for i, (hi, hi_end, wi, wi_end) in enumerate(tiles_iterator):
                # Use different colors for different tiles
                color = [
                    int(255 * ((i * 137) % 256) / 256),  # Red
                    int(255 * ((i * 211) % 256) / 256),  # Green
                    int(255 * ((i * 101) % 256) / 256)  # Blue
                ]

                # Draw tile boundaries
                # Top and bottom lines
                vis_image[hi:hi + 3, wi:wi_end] = color
                vis_image[max(0, hi_end - 3):hi_end, wi:wi_end] = color
                # Left and right lines
                vis_image[hi:hi_end, wi:wi + 3] = color
                vis_image[hi:hi_end, max(0, wi_end - 3):wi_end] = color

            return vis_image, tile_count

        return input_image, tile_count

    def process_with_tiling(self, input_image: np.ndarray,
                            prompts: List[str],
                            upscale_factor: int = 2,
                            num_steps: int = 50,
                            guidance_scale: float = 7.5,
                            control_scale: float = 1.0,
                            seed: int = -1,
                            color_fix_type: str = "Wavelet",
                            use_llava_per_tile: bool = False,
                            temperature: float = 0.7,
                            top_p: float = 0.9,
                            custom_prompt: str = None) -> Dict[str, Any]:
        """Process image using tiled approach for VRAM efficiency"""

        if input_image is None:
            return {"error": "No input image provided"}

        torch.cuda.set_device(self.imageai_device)

        try:
            # Prepare input image
            input_image = HWC3(input_image)
            if upscale_factor > 1:
                input_image = upscale_image(input_image, upscale_factor, unit_resolution=32, min_size=1024)

            # Convert to tensor
            input_tensor = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(
                self.imageai_device) / 255.0 * 2 - 1

            # Enable tiled VAE if available
            if self.args.use_tile_vae and hasattr(self.model, 'init_tile_vae'):
                self.model.init_tile_vae(
                    encoder_tile_size=self.args.encoder_tile_size,
                    decoder_tile_size=self.args.decoder_tile_size
                )

            # Process with tiled sampling if available and enabled
            if self.enable_tiling and hasattr(self.model, 'sample_with_tiling'):
                # Use model's built-in tiling
                samples = self.model.sample_with_tiling(
                    input_image=input_tensor,
                    prompts=prompts,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    control_scale=control_scale,
                    seed=seed,
                    color_fix_type=color_fix_type,
                    tile_size=self.tile_size,
                    tile_stride=self.tile_stride
                )
            else:
                # Standard processing
                samples = self.model.sample(
                    input_image=input_tensor,
                    prompts=prompts,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    control_scale=control_scale,
                    seed=seed,
                    color_fix_type=color_fix_type
                )

            # Convert results
            if len(samples.shape) == 4:  # Batch of images
                results = []
                for i in range(samples.shape[0]):
                    img = (samples[i].permute(1, 2, 0) * 255).cpu().numpy().round().clip(0, 255).astype(np.uint8)
                    results.append(img)
            else:  # Single image
                img = (samples.permute(1, 2, 0) * 255).cpu().numpy().round().clip(0, 255).astype(np.uint8)
                results = [img]

            # Optional tile analysis with LLaVA
            tile_info = {}
            tile_captions = []

            if use_llava_per_tile and self.use_llava:
                h, w = input_image.shape[:2]
                image_tile_size = self.tile_size * 8
                image_tile_stride = self.tile_stride * 8

                tiles_iterator = self._sliding_windows(h, w, image_tile_size, image_tile_stride)

                for i, (hi, hi_end, wi, wi_end) in enumerate(tiles_iterator):
                    tile_img = input_image[hi:hi_end, wi:wi_end]

                    llava_prompt = custom_prompt if custom_prompt else "Describe what you see in this image tile."
                    caption = self.llava_caption_process(tile_img, temperature, top_p, llava_prompt)
                    tile_captions.append(f"Tile {i + 1}: {caption}")

                tile_info = {
                    "tile_count": len(list(tiles_iterator)),
                    "tile_captions": tile_captions,
                    "tile_size": f"{self.tile_size} (latent) / ~{image_tile_size}px (image)",
                    "tile_stride": f"{self.tile_stride} (latent) / ~{image_tile_stride}px (image)"
                }

            return {
                "processed_images": results,
                "tile_info": tile_info,
                "success": True
            }

        except Exception as e:
            error_msg = f"Error in tiled processing: {str(e)}"
            print(error_msg)
            return {"error": error_msg, "success": False}

    @staticmethod
    def fix_dimensions(image):
        """Fix image dimensions for PhotoAI compatibility"""
        if isinstance(image, np.ndarray):
            # Convert numpy to PIL for resizing
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        w, h = pil_image.size

        # Round up to nearest multiple of 16
        new_w = ((w + 15) // 16) * 16
        new_h = ((h + 15) // 16) * 16

        if new_w != w or new_h != h:
            print(f"📐 Resizing for tensor compatibility: {w}×{h} → {new_w}×{new_h}")
            pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Convert back to numpy if that's what was passed in
        if isinstance(image, np.ndarray):
            return np.array(pil_image)

        return pil_image

    def create_common_interface_components(self):
        """Create common Gradio interface components used across all demos"""

        # Input components
        input_image = gr.Image(label="Input Image", type="numpy")

        # Basic processing controls
        with gr.Accordion("Basic Settings", open=True):
            with gr.Row():
                upscale_factor = gr.Slider(1, 4, value=2, step=1, label="Upscale Factor")
                num_steps = gr.Slider(10, 100, value=50, step=5, label="Sampling Steps")

            with gr.Row():
                guidance_scale = gr.Slider(1.0, 15.0, value=7.5, label="CFG Scale")
                control_scale = gr.Slider(0.0, 2.0, value=1.0, label="Control Scale")

            with gr.Row():
                seed = gr.Number(label="Seed", value=-1, precision=0)
                color_fix_type = gr.Dropdown(
                    choices=["Wavelet", "AdaIn", "None"],
                    value="Wavelet",
                    label="Color Fix Type"
                )

        # Tiling options
        with gr.Accordion("Tiling Options (VRAM Efficiency)", open=False):
            enable_tiling = gr.Checkbox(
                value=self.enable_tiling,
                label="Enable Tiled Processing",
                info="Process image in tiles to save VRAM for large images"
            )

            with gr.Row():
                tile_size = gr.Slider(
                    32, 256, value=self.tile_size, step=16,
                    label="Tile Size (Latent)",
                    info="Larger tiles = better quality, more VRAM"
                )
                tile_stride = gr.Slider(
                    16, 128, value=self.tile_stride, step=8,
                    label="Tile Stride (Latent)",
                    info="Smaller stride = more overlap, better blending"
                )

            gr.Markdown("*Note: Sizes are in latent space. Multiply by ~8 for image space.*")

        # LLaVA options
        llava_accordion = gr.Accordion("LLaVA Analysis", open=False)
        with llava_accordion:
            prompt = gr.Textbox(
                label="Generation Prompt (optional)",
                placeholder="Leave empty for enhancement only or describe desired changes",
                lines=2
            )

            with gr.Row():
                temperature = gr.Slider(0.1, 1.0, value=0.7, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, label="Top P")

            custom_prompt = gr.Textbox(
                label="Custom Analysis Prompt",
                value="Describe what you see in this image tile.",
                lines=2
            )

            use_llava_per_tile = gr.Checkbox(
                value=False,
                label="Analyze Each Tile with LLaVA",
                info="Generate captions for each tile (slower but detailed)"
            )

        # Disable LLaVA components if not available
        if not self.use_llava:
            llava_accordion.visible = False

        return {
            'input_image': input_image,
            'upscale_factor': upscale_factor,
            'num_steps': num_steps,
            'guidance_scale': guidance_scale,
            'control_scale': control_scale,
            'seed': seed,
            'color_fix_type': color_fix_type,
            'enable_tiling': enable_tiling,
            'tile_size': tile_size,
            'tile_stride': tile_stride,
            'prompt': prompt,
            'temperature': temperature,
            'top_p': top_p,
            'custom_prompt': custom_prompt,
            'use_llava_per_tile': use_llava_per_tile
        }

    def create_model_info_accordion(self):
        """Create model information accordion"""
        with gr.Accordion("Model Information", open=False):
            model_info = self.get_model_info()
            info_md = f"""
            **ImageAI Device:** {model_info['imageai_device']}  
            **LLaVA Device:** {model_info['llava_device']}  
            **LLaVA Model:** {model_info['llava_model']}  
            **LLaVA Status:** {model_info['llava_status']}  
            **Model Type:** {model_info['model_type']}  
            **Scheduler:** {model_info['scheduler']}  
            **VAE Type:** {model_info['vae_type']}  
            **Model Variant:** {getattr(self.args, 'model_variant', 'default')}

            **Optimizations Enabled:**
            - XFormers: {'✅' if getattr(self.args, 'enable_xformers', False) else '❌'}
            - CPU Offload: {'✅' if getattr(self.args, 'enable_cpu_offload', False) else '❌'}
            - Model Compile: {'✅' if getattr(self.args, 'compile_model', False) else '❌'}
            - Half Precision: {'✅' if getattr(self.args, 'loading_half_params', False) else '❌'}
            - Tiled VAE: {'✅' if self.args.use_tile_vae else '❌'}
            - Tiled Processing: {'✅' if self.enable_tiling else '❌'}

            **Tiling Configuration:**
            - Tile Size: {self.tile_size} (latent) / ~{self.tile_size * 8}px (image)
            - Tile Stride: {self.tile_stride} (latent) / ~{self.tile_stride * 8}px (image)
            - Tile Overlap: {self.tile_size - self.tile_stride} (latent) / ~{(self.tile_size - self.tile_stride) * 8}px (image)
            """
            gr.Markdown(info_md)

    def log_event(self, event_dict: Dict[str, Any]):
        """Log processing events if enabled"""
        if self.args.log_history:
            try:
                with open("logs.jsonl", "a") as f:
                    f.write(json.dumps(event_dict) + "\n")
            except Exception as e:
                print(f"Warning: Failed to log event: {e}")

    @staticmethod
    def load_config_and_create_model(config_path: str, device: str = None):
        """Load config and create PhotoAI model with proper component extraction"""
        print(f"Loading config from: {config_path}")

        # Create PhotoAI model using the actual function signature
        model = create_photoai_model(
            config_path=config_path,
            photoai_sign='Q',
            load_default_setting=False
        )

        # Move to device if specified
        if device:
            model = model.to(device)

        return model

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for display - adapted for PhotoAI's SUPIRModel"""
        scheduler_name = "Unknown"
        vae_name = "Unknown"

        try:
            # Try to get scheduler info (PhotoAI uses sampler)
            if self.model and hasattr(self.model, 'sampler'):
                scheduler_name = self.model.sampler.__class__.__name__
            elif self.model and hasattr(self.model, 'sampler_config'):
                # Try to get from config
                scheduler_name = "EDMEulerSampler (configured)"
        except:
            scheduler_name = "Unknown"

        try:
            # Try to get VAE info
            if self.model and hasattr(self.model, 'first_stage_model'):
                vae_name = self.model.first_stage_model.__class__.__name__
        except:
            vae_name = "Unknown"

        return {
            "imageai_device": self.imageai_device,
            "llava_device": self.llava_device,
            "llava_model": self.args.llava_model if self.use_llava else 'Disabled',
            "llava_status": '✅ Loaded' if self.use_llava else '❌ Disabled',
            "model_type": "Erik''s Photo Model",
            "scheduler": scheduler_name,
            "vae_type": vae_name
        }

    @abstractmethod
    def setup_gradio_interface(self):
        """Setup Gradio interface - to be implemented by subclasses"""
        pass

    @abstractmethod
    def launch(self):
        """Launch the demo - to be implemented by subclasses"""
        pass


def create_base_parser() -> argparse.ArgumentParser:
    """Create base argument parser with enhanced options"""
    parser = argparse.ArgumentParser()

    # Server options
    parser.add_argument("--ip", type=str, default='127.0.0.1', help="Server IP address")
    parser.add_argument("--port", type=int, default=6688, help="Server port")

    # Model options
    parser.add_argument("--model_variant", type=str, default='base',
                        choices=['base', 'lightning', 'tiled'],
                        help="Model variant to use")
    parser.add_argument("--loading_half_params", action='store_true', default=False,
                        help="Load model in half precision")

    # LLaVA options
    parser.add_argument("--no_llava", action='store_true', default=False,
                        help="Disable LLaVA model")
    parser.add_argument("--llava_model", type=str,
                        default="llava-hf/llava-v1.6-mistral-7b-hf",
                        help="LLaVA model name or path")
    parser.add_argument("--load_8bit_llava", action='store_true', default=False,
                        help="Load LLaVA in 8-bit mode")
    parser.add_argument("--load_4bit_llava", action='store_true', default=False,
                        help="Load LLaVA in 4-bit mode")

    # Optimization options
    parser.add_argument("--enable_xformers", action='store_true', default=False,
                        help="Enable xformers memory efficient attention")
    parser.add_argument("--enable_cpu_offload", action='store_true', default=False,
                        help="Enable CPU offloading to save VRAM")
    parser.add_argument("--compile_model", action='store_true', default=False,
                        help="Compile model with torch.compile")

    # Tiled VAE options
    parser.add_argument("--use_tile_vae", action='store_true', default=False,
                        help="Enable tiled VAE processing")
    parser.add_argument("--encoder_tile_size", type=int, default=512,
                        help="VAE encoder tile size")
    parser.add_argument("--decoder_tile_size", type=int, default=64,
                        help="VAE decoder tile size")

    # Tiled processing options
    parser.add_argument("--enable_tiling", action='store_true', default=False,
                        help="Enable tiled diffusion processing")
    parser.add_argument("--tile_size", type=int, default=128,
                        help="Diffusion tile size in latent space")
    parser.add_argument("--tile_stride", type=int, default=64,
                        help="Diffusion tile stride in latent space")

    # Utility options
    parser.add_argument("--use_image_slider", action='store_true', default=False,
                        help="Use image slider for before/after comparison")
    parser.add_argument("--log_history", action='store_true', default=False,
                        help="Log processing history")
    parser.add_argument("--local_prompt", action='store_true', default=False,
                        help="Use local prompting")

    return parser


def get_config_path(variant: str) -> str:
    """Get config path based on variant - using actual PhotoAI config names"""
    config_map = {
        'base': 'options/PhotoAI.yaml',  # Using actual PhotoAI config names
        'lightning': 'options/PhotoAI_lightning.yaml',
        'tiled': 'options/PhotoAI_tiled.yaml'
    }
    return config_map.get(variant, config_map['base'])


def setup_model_optimizations(model, args):
    """Setup model optimizations based on arguments - adapted for PhotoAI"""

    # Enable xformers if requested (PhotoAI uses different structure)
    if getattr(args, 'enable_xformers', False):
        try:
            # PhotoAI may have xformers in the model.model (UNet)
            if hasattr(model, 'model') and hasattr(model.model, 'enable_xformers_memory_efficient_attention'):
                model.model.enable_xformers_memory_efficient_attention()
                print("✅ Enabled xformers memory efficient attention")
            else:
                print("⚠️ XFormers not available on this model")
        except Exception as e:
            print(f"⚠️ Failed to enable xformers: {e}")

    # Enable CPU offload if requested (may not be applicable to model)
    if getattr(args, 'enable_cpu_offload', False):
        try:
            if hasattr(model, 'enable_model_cpu_offload'):
                model.enable_model_cpu_offload()
                print("✅ Enabled CPU offloading")
            else:
                print("⚠️ CPU offload not available on model")
        except Exception as e:
            print(f"⚠️ Failed to enable CPU offload: {e}")

    # Compile model if requested
    if getattr(args, 'compile_model', False):
        try:
            if hasattr(model, 'model'):
                model.model = torch.compile(model.model, mode="reduce-overhead")
                print("✅ Compiled model with torch.compile")
            else:
                print("⚠️ Model compilation not available")
        except Exception as e:
            print(f"⚠️ Failed to compile model: {e}")

    # Set precision (PhotoAI already handles this via ae_dtype and diffusion_dtype)
    if getattr(args, 'loading_half_params', False):
        try:
            # PhotoAI handles precision via ae_dtype and model.dtype
            if hasattr(model, 'ae_dtype'):
                print(
                    f"✅ PhotoAI model precision: AE={model.ae_dtype}, Model={getattr(model.model, 'dtype', 'Unknown')}")
            else:
                model = model.half()
                print("✅ Set model to half precision")
        except Exception as e:
            print(f"⚠️ Failed to set precision: {e}")

    # Setup tiled VAE if requested (PhotoAI specific method)
    if getattr(args, 'use_tile_vae', False):
        try:
            if hasattr(model, 'init_tile_vae'):
                model.init_tile_vae(
                    encoder_tile_size=getattr(args, 'encoder_tile_size', 512),
                    decoder_tile_size=getattr(args, 'decoder_tile_size', 64)
                )
                print(
                    f"✅ Enabled tiled VAE (encoder: {getattr(args, 'encoder_tile_size', 512)}, decoder: {getattr(args, 'decoder_tile_size', 64)})")
            else:
                print("⚠️ Tiled VAE not available on this model")
        except Exception as e:
            print(f"⚠️ Failed to enable tiled VAE: {e}")

    return model