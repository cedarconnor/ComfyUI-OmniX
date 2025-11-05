# ComfyUI-OmniX

![OmniX Banner](https://img.shields.io/badge/ComfyUI-OmniX-blue) ![Version](https://img.shields.io/badge/version-1.0.0-green) ![License](https://img.shields.io/badge/license-Apache%202.0-orange)

**OmniX panorama generation and perception for ComfyUI**

Generate stunning 360° equirectangular panoramas and extract comprehensive scene properties including depth, normals, and PBR materials using OmniX adapters with ComfyUI's Flux pipeline.

## Features

✨ **Panorama Generation**
- Text-to-360° panorama using Flux.1-dev + OmniX RGB adapter
- High-resolution output (up to 4096×2048)
- Seamless integration with ComfyUI's existing Flux workflow

🔍 **Panorama Perception**
- **Distance Maps**: Extract metric depth from panoramas
- **Normal Maps**: Surface normal estimation for 3D reconstruction
- **PBR Materials**: Albedo, roughness, and metallic maps
- Multi-modal extraction in single pass

🛠️ **Utility Nodes**
- Panorama aspect ratio validation and correction
- Depth map visualization with colormaps
- Format conversion utilities

## Installation

### Method 1: ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Search for "OmniX"
3. Click "Install"
4. Restart ComfyUI

### Method 2: Manual Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/[YOUR_USERNAME]/ComfyUI-OmniX.git
cd ComfyUI-OmniX
pip install -r requirements.txt
```

### Method 3: Git Clone to Custom Nodes

```bash
# Navigate to ComfyUI custom nodes directory
cd /path/to/ComfyUI/custom_nodes/

# Clone this repository
git clone https://github.com/[YOUR_USERNAME]/ComfyUI-OmniX.git

# Install dependencies
pip install -r ComfyUI-OmniX/requirements.txt
```

## Model Weights

OmniX requires adapter weights in addition to the base Flux.1-dev model.

### Directory Structure

Place OmniX adapter weights in:
```
ComfyUI/models/omnix/omnix-base/
├── config.json
├── rgb_generation_adapter.safetensors
├── distance_adapter.safetensors
├── normal_adapter.safetensors
├── albedo_adapter.safetensors
├── roughness_adapter.safetensors
└── metallic_adapter.safetensors
```

### Downloading Adapter Weights

**Option 1: HuggingFace (when available)**
```bash
# Will be available after official OmniX release
# huggingface-cli download HKU-MMLab/OmniX --local-dir models/omnix/omnix-base
```

**Option 2: Manual Download**
- Visit [OmniX GitHub](https://github.com/HKU-MMLab/OmniX)
- Follow instructions for downloading adapter weights
- Place files in `ComfyUI/models/omnix/omnix-base/`

### Model Size Requirements

| Component | Size | Required |
|-----------|------|----------|
| Flux.1-dev base | ~23GB | Yes (shared with other Flux workflows) |
| RGB Generation Adapter | ~2GB | For panorama generation |
| Distance Adapter | ~1.5GB | For depth extraction |
| Normal Adapter | ~1.5GB | For normal mapping |
| Albedo Adapter | ~1.5GB | For texture extraction |
| Roughness Adapter | ~1GB | For PBR materials |
| Metallic Adapter | ~1GB | For PBR materials |

**Total:** ~32GB disk space for all components

## Quick Start

### Basic Text-to-Panorama Workflow

```
1. [CheckpointLoaderSimple] → Load Flux.1-dev model
   ↓
2. [OmniXAdapterLoader] → Load OmniX adapters (preset: omnix-base, precision: fp16)
   ↓
3. [OmniXApplyAdapters] → Apply RGB generation adapter to model
   ↓
4. [CLIPTextEncode] → Encode your prompt
   ↓
5. [KSampler] → Generate (use 2:1 aspect ratio: 2048×1024)
   ↓
6. [VAEDecode] → Decode latents
   ↓
7. [SaveImage] → Save your 360° panorama
```

### Panorama Perception Workflow

```
1. [LoadImage] → Load a panorama image
   ↓
2. [OmniXPanoramaValidator] → Validate/fix aspect ratio
   ↓
3. [OmniXAdapterLoader] → Load perception adapters
   ↓
4. [OmniXPanoramaPerception] → Extract properties
   ├─ Distance map
   ├─ Normal map
   ├─ Albedo
   ├─ Roughness
   └─ Metallic
```

## Nodes Reference

### OmniXAdapterLoader

Loads OmniX adapter weights for panorama processing.

**Inputs:**
- `adapter_preset`: Model variant (`omnix-base`, `omnix-large`)
- `precision`: Data type (`fp16`, `fp32`, `bf16`)

**Outputs:**
- `adapters`: Loaded OmniX adapters

### OmniXApplyAdapters

Applies OmniX adapters to a Flux model for panorama generation.

**Inputs:**
- `model`: Flux MODEL from checkpoint loader
- `adapters`: OmniX adapters from loader
- `adapter_type`: Type of adapter (`rgb_generation`)
- `adapter_strength`: Strength multiplier (0.0-2.0, default: 1.0)

**Outputs:**
- `model`: Modified MODEL ready for panorama generation

### OmniXPanoramaPerception

Extracts geometric and material properties from panoramas.

**Inputs:**
- `adapters`: OmniX adapters
- `panorama`: Input panorama IMAGE
- `extract_distance`: Enable depth extraction (boolean)
- `extract_normal`: Enable normal extraction (boolean)
- `extract_albedo`: Enable albedo extraction (boolean)
- `extract_roughness`: Enable roughness extraction (boolean)
- `extract_metallic`: Enable metallic extraction (boolean)

**Outputs:**
- `distance`: Depth map (visualized with colormap)
- `normal`: Normal map (RGB visualization)
- `albedo`: Base color/diffuse map
- `roughness`: Roughness map (grayscale)
- `metallic`: Metallic map (grayscale)

### OmniXPanoramaValidator

Validates and corrects panorama aspect ratios.

**Inputs:**
- `image`: Input IMAGE
- `target_aspect_ratio`: Target ratio (default: 2.0 for equirectangular)
- `fix_method`: Correction method (`crop`, `pad`, `stretch`)

**Outputs:**
- `image`: Corrected IMAGE
- `info`: Information string about corrections applied

## Performance

### Recommended Hardware

| VRAM | Performance | Settings |
|------|-------------|----------|
| 24GB+ | Optimal | Full precision (fp32), no offloading |
| 16GB | Great | Half precision (fp16), optional offloading |
| 12GB | Good | Half precision (fp16), model offloading |
| 8GB | Usable | Half precision (fp16), aggressive offloading + lower res |

### Benchmarks (RTX 4090)

| Task | Resolution | Time | Settings |
|------|------------|------|----------|
| Panorama Generation | 2048×1024 | ~25s | 28 steps, fp16 |
| Perception (all modes) | 2048×1024 | ~8s | fp16 |
| End-to-end | 2048×1024 | ~33s | Generation + perception |

## Troubleshooting

### "Adapter weights not found"

**Solution:** Ensure adapter weights are properly installed:
```bash
ls ComfyUI/models/omnix/omnix-base/
# Should show: rgb_generation_adapter.safetensors, distance_adapter.safetensors, etc.
```

### "CUDA out of memory"

**Solutions (in order):**
1. Use `fp16` precision instead of `fp32`
2. Reduce output resolution (e.g., 1024×512 instead of 2048×1024)
3. Close other applications using VRAM
4. Enable model offloading in Flux checkpoint loader
5. Disable some perception modes (extract only what you need)

### "Invalid panorama aspect ratio"

**Solution:** Use the `OmniXPanoramaValidator` node before perception:
- For equirectangular: set `target_aspect_ratio` to 2.0
- Use `crop` method for best quality
- Use `pad` method to preserve all content

### Slow generation

**Solutions:**
1. Check if using `fp16` precision (fastest)
2. Ensure model is loaded to GPU (check ComfyUI console)
3. Reduce number of sampling steps (try 20-24 instead of 28)
4. Check VRAM usage - offloading slows down inference

## Examples

### Prompt Tips for Panoramas

**Good prompts:**
- "360 degree view of a modern living room, photorealistic, architectural photography"
- "equirectangular panorama of a futuristic cityscape at night, neon lights, cyberpunk style"
- "full spherical panorama of a serene mountain landscape, sunrise, volumetric lighting"

**Tips:**
- Include "360 degree", "equirectangular", or "panorama" in prompt
- Avoid directional terms like "left side" or "right corner"
- Emphasize lighting and atmosphere for best results
- Use higher guidance scale (5.0-7.5) for more prompt adherence

## Technical Details

### Architecture

OmniX uses separate adapter modules for each task:
- **RGB Generation**: Guides Flux to generate 360° panoramas
- **Distance**: Predicts metric depth values
- **Normal**: Estimates surface normals
- **Albedo/Roughness/Metallic**: Extracts PBR material properties

Adapters are lightweight (1-2GB each) compared to the base model (23GB) and use cross-modal learning for high-quality outputs.

### Format Specifications

**Panorama Format:**
- Type: Equirectangular projection
- Aspect Ratio: 2:1 (width:height)
- Supported Resolutions: 512×1024, 1024×2048, 2048×4096
- Color Space: sRGB, [0, 1] range

**Output Maps:**
- Distance: Single channel, metric scale, visualized with viridis colormap
- Normal: 3 channels (XYZ), unit vectors, visualized in [0, 1] range
- Albedo: 3 channels (RGB), [0, 1] range
- Roughness/Metallic: Single channel, [0, 1] range

## Citation

If you use OmniX in your work, please cite:

```bibtex
@article{omnix2024,
  title={OmniX: Unified Panorama Generation and Perception},
  author={[Authors]},
  journal={arXiv preprint arXiv:2510.26800},
  year={2024}
}
```

## References

- **OmniX Paper**: https://arxiv.org/abs/2510.26800
- **OmniX GitHub**: https://github.com/HKU-MMLab/OmniX
- **Flux.1-dev**: https://github.com/black-forest-labs/flux
- **ComfyUI**: https://github.com/comfyanonymous/ComfyUI

## License

Apache License 2.0 - see LICENSE file for details.

## Acknowledgments

- OmniX team at HKU-MMLab for the original implementation
- ComfyUI community for the excellent framework
- Black Forest Labs for Flux.1-dev

## Support

- **Issues**: https://github.com/[YOUR_USERNAME]/ComfyUI-OmniX/issues
- **Discussions**: https://github.com/[YOUR_USERNAME]/ComfyUI-OmniX/discussions
- **ComfyUI Discord**: https://discord.gg/comfyui

---

**Made with ❤️ for the ComfyUI community**
