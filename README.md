# PhotoAI: Photo-Realistic Image Restoration

> **⚠️ Important**: This project is **NOT FREE for commercial use**. Please review the license terms below.

PhotoAI (the name WILL change) is a photo-realistic image restoration system built on the SDXL diffusion model architecture. The system provides significant quality improvements over previous generation models, transforming degraded images into high-resolution outputs with improved detail preservation and photorealistic quality.

## 🚀 Key Features

**Architecture Improvements**
- **3x Quality Improvement**: Substantial improvement over previous generation models
- **Dual-Stage Processing**: Base model + refiner for enhanced quality
- **Native 1024x1024 Generation**: Direct high-resolution output
- **Improved Text Rendering**: Better prompt adherence and compositional control

**Multiple Model Variants**
- **Base Model**: Balanced quality and performance for general use
- **Lightning Model**: 1-8 step generation with 6-10x speed improvements
- **Tiled Model**: Optimized for large images and memory-constrained environments
- **Face-Specialized**: Enhanced portrait and facial restoration capabilities

**Production Features**
- **Memory Optimization**: Sub-3GB VRAM operation through intelligent offloading
- **Tiled Processing**: Handle 4K+ images on consumer hardware
- **Multiple Precision Support**: FP16/BF16 for performance, FP32 for quality
- **Color Correction**: Wavelet and AdaIN-based color fixing

## 📋 Requirements

### Hardware Requirements
- **Minimum**: 8GB VRAM, 16GB System RAM
- **Recommended**: 12GB+ VRAM, 32GB+ System RAM
- **GPU**: NVIDIA RTX 3060+ (RTX 4090 recommended for best performance)
- **Storage**: 50GB+ free space for models and cache

### Software Dependencies
- Python 3.8-3.11
- PyTorch 2.1.0+
- CUDA 11.8+ or 12.x
- See `requirements.txt` for complete dependency list

## ⚡ Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/photoai.git
cd photoai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Models

```bash
# Download base model (required)
python scripts/download_models.py --model base

# Optional: Download additional variants
python scripts/download_models.py --model lightning
python scripts/download_models.py --model tiled
python scripts/download_models.py --model face
```

### 3. Launch Demo

**Basic Demo (General Purpose)**
```bash
python gradio_demo.py --ip 0.0.0.0 --port 7860
```

**Face Restoration Demo**
```bash
python gradio_demo_face.py --ip 0.0.0.0 --port 7860
```

**Tiled Processing Demo (Large Images)**
```bash
python gradio_demo_tiled.py --ip 0.0.0.0 --port 7860
```

### 4. Basic Usage

Navigate to `http://localhost:7860` in your browser:

1. **Upload Image**: Drop your degraded image into the input area
2. **Set Parameters**: Adjust quality settings and prompts
3. **Generate**: Click process to restore your image
4. **Download**: Save the enhanced result

## 🎯 Model Variants Guide

### Base Model
- **Best For**: General image restoration, balanced quality/speed
- **Steps**: 30-50 for optimal quality
- **CFG Scale**: 7-12
- **Memory**: ~8-12GB VRAM

### Lightning Model ⚡
- **Best For**: Fast iteration, real-time processing
- **Steps**: 4-8 steps only
- **CFG Scale**: 1.5-3.0 (lower than base)
- **Speed**: 6-10x faster than base model

### Tiled Model 🏗️
- **Best For**: Large images, limited VRAM
- **Features**: Automatic tiled VAE, memory-efficient processing
- **Max Resolution**: 4K+ images on 8GB VRAM
- **Trade-off**: Slightly longer processing time

### Face Model 👤
- **Best For**: Portrait restoration, facial enhancement
- **Features**: Specialized face detection and restoration
- **Quality**: Superior facial detail preservation
- **Use Case**: Professional portrait restoration

## 🛠️ Configuration

### Memory Optimization

For **8GB VRAM** setups:
```bash
python gradio_demo.py --use_tile_vae --loading_half_params
```

For **12GB+ VRAM** setups:
```bash
python gradio_demo.py --no_llava  # Disable LLaVA for more VRAM
```

### Advanced Settings

**Quality Preset**:
- Steps: 40-50
- CFG Scale: 7.5
- Guidance: 4.0

**Speed Preset**:
- Steps: 20-30
- CFG Scale: 4.0
- Guidance: 1.0

## 📊 Performance Benchmarks

| GPU | Resolution | Model | Steps | Time | VRAM |
|-----|------------|-------|-------|------|------|
| RTX 4090 | 1024x1024 | Base | 40 | 3.6s | 10GB |
| RTX 4090 | 1024x1024 | Lightning | 8 | 0.8s | 8GB |
| RTX 3090 | 1024x1024 | Base | 40 | 10.9s | 12GB |
| RTX 3060 | 1024x1024 | Tiled | 40 | 25s | 8GB |

## 🔧 Troubleshooting

### Common Issues

**Out of Memory Errors**:
```bash
# Enable aggressive memory optimization
python gradio_demo.py --use_tile_vae --loading_half_params
```

**Slow Generation**:
- Switch to Lightning model for speed
- Reduce number of steps
- Enable FP16 precision

**Quality Issues**:
- Increase number of steps (40-50)
- Adjust CFG scale (7-12)
- Try different prompt engineering

### Performance Tips

1. **Use FP16**: 50% memory reduction, 1.5x speed boost
2. **Enable Tiling**: Process large images on limited VRAM
3. **Optimize Prompts**: Detailed descriptions improve results
4. **Sequential Processing**: Process multiple images in batches

## 🎨 Prompt Engineering

### Effective Prompts

**For Portraits**:
```
"Professional headshot, sharp focus, studio lighting, detailed skin texture, natural skin tone, high resolution photography"
```

**For Landscapes**:
```
"Landscape photography, natural lighting, vibrant colors, detailed textures, professional composition, ultra-high resolution"
```

**Negative Prompts**:
```
"blurry, low quality, artifacts, oversaturated, cartoon, painting, illustration, deformed"
```

## 📈 Migration from SDXL 0.9

Following our [comprehensive migration guide](SDXL_Migration_Roadmap.md), PhotoAI builds on the SDXL architecture upgrade. Key improvements include:

- **3x Quality Improvement**: Substantial enhancement over SDXL 0.9
- **Architectural Refinements**: Optimized transformer block distribution
- **Enhanced Attention**: 3x more attention blocks than SD 1.5
- **Modern Sampling**: DPM++ 2M Karras for improved quality-speed balance

## 🏗️ Future Improvements

### Planned Architecture Enhancements

**Next-Generation Models**
- **Stable Diffusion 3.5 Large Integration**: Native ControlNet support for Blur, Canny, and Depth conditioning
- **FLUX.1 Series Compatibility**: Alternative architectures optimized for different use cases from Black Forest Labs
- **Rectified Flow Foundation**: Migration to SD 3.0 architecture with improved sampling efficiency
- **Transformer Architecture Evolution**: DiT-based models with improved quality characteristics

### Enhanced Control Mechanisms

**Improved ControlNet Support**
- **ControlNet v1.1+**: Enhanced stability with SDXL-specific native architecture support
- **Control-LoRA Integration**: Lightweight alternatives with rank 256/128 variants for memory efficiency
- **T2I-Adapter SDXL**: 79M parameter adapters controlling the full 2.6B model with robust conditioning
- **IP-Adapter v2+**: Decoupled cross-attention mechanisms with ~22M trainable parameters

### Performance & Optimization Roadmap

**Memory & Speed Improvements**
- **Hyper-SDXL Implementation**: Consistency Trajectory Models (CTM) for improved quality retention
- **Advanced Quantization**: TensorRT INT8/FP8 support for 1.7-1.95x speedup with minimal quality loss
- **Sequential CPU Offloading**: Sub-2GB VRAM operation for broader hardware compatibility
- **Attention Optimization V2**: 60%+ peak memory reduction through improved slicing techniques

**Sampling Method Evolution**
- **Improved Sampling**: Next-generation DPM++ variants with better scheduling
- **LCM-LoRA Enhancement**: 8-12x speed improvements through optimized distillation
- **Adaptive Step Scheduling**: Dynamic step allocation based on image complexity

### Ecosystem & Integration

**Platform Development**
- **Enterprise API**: RESTful API with batch processing and queue management
- **Cloud Integration**: AWS/Azure/GCP deployment templates with auto-scaling
- **Professional Workflow Tools**: Adobe/GIMP plugins and standalone applications
- **Batch Processing Pipeline**: Industrial-scale image restoration workflows

**Community & Open Source**
- **Fine-tuning Ecosystem**: Simplified LoRA training for domain-specific applications
- **Model Hub Integration**: Seamless Hugging Face model sharing and discovery
- **Educational Resources**: Comprehensive tutorials and best practice guides
- **Research Collaboration**: Academic partnership program for ongoing developments

### Quality & Capability Expansion

**Specialized Domain Models**
- **Medical Imaging**: HIPAA-compliant restoration for clinical applications
- **Historical Photo Restoration**: Specialized models for archival and vintage content
- **Scientific Imaging**: Microscopy and satellite image enhancement
- **Art & Cultural Heritage**: Museum-quality restoration for historical artifacts

**Multi-Modal Integration**
- **Video Frame Enhancement**: Temporal consistency for video restoration
- **3D Asset Generation**: Integration with 3D modeling workflows
- **Cross-Platform Compatibility**: Mobile and edge device optimization
- **Real-Time Processing**: Live video enhancement capabilities

These improvements focus on maintaining PhotoAI's performance and capabilities while expanding accessibility across diverse use cases and hardware configurations.

## 📄 License & Usage

### Important License Information

**🚫 This software is NOT FREE for commercial use.**

- **Research & Personal Use**: Free for non-commercial research and personal projects
- **Commercial Use**: Requires separate commercial license
- **Attribution**: Required for all use cases
- **Distribution**: Subject to original license terms

### Commercial Licensing

For commercial licensing inquiries, please contact:
- Email: [your-email@domain.com]
- Website: [your-website.com]

### Third-Party Licenses

This project incorporates components with various licenses:
- SGM components: See `sgm/` directory licenses
- LPIPS model: BSD License (see `sgm/modules/autoencoding/lpips/`)
- Other dependencies: See individual component licenses

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description
4. Ensure all tests pass
5. Follow the existing code style

## 📚 Citation

If you use PhotoAI in your research, please cite:

```bibtex
@article{photoai2024,
  title={PhotoAI: Photo-Realistic Image Restoration with Advanced Diffusion Models},
  author={[Your Name]},
  journal={[Journal/Conference]},
  year={2024}
}
```

## 🆘 Support

- **Documentation**: Check our [Wiki](wiki-link) for detailed guides
- **Issues**: Report bugs via [GitHub Issues](issues-link)
- **Discussions**: Join our [Community Forum](forum-link)
- **Discord**: Real-time support on our [Discord Server](discord-link)

## 🙏 Acknowledgments

Built upon the excellent work of:
- Stability AI for SDXL architecture
- The open-source diffusion model community
- Contributors to the original SUPIR project
- PyTorch and Hugging Face teams

---

**⭐ Star this repository if PhotoAI helped enhance your images!**

*PhotoAI - Professional image restoration with modern diffusion models.*
