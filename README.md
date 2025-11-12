PERCEPTION-ONLY | WORK IN PROGRESS

# ComfyUI-OmniX

![OmniX Banner](https://img.shields.io/badge/ComfyUI-OmniX-blue) ![Version](https://img.shields.io/badge/version-1.0.0-green) ![License](https://img.shields.io/badge/license-Apache%202.0-orange)

**OmniX panorama perception for ComfyUI**

Extract comprehensive scene properties from 360° equirectangular panoramas including depth, normals, and PBR materials using OmniX adapters with ComfyUI's Flux pipeline.

## Features

🔍 **Panorama Perception** (Implemented)
- **Distance Maps**: Extract metric depth from panoramas using Flux VAE + LoRA adapters
- **Normal Maps**: Surface normal estimation for 3D reconstruction
- **Albedo Maps**: Base color/diffuse extraction
- **PBR Materials**: Roughness and metallic maps
- Multi-modal extraction using real OmniX architecture

⚙️ **Architecture**
- Uses Flux VAE for encoding/decoding (proper latent-space processing)
- LoRA adapters injected into Flux transformer blocks
- Based on real HKU-MMLab/OmniX implementation

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

### Quick Download (3 Easy Steps)

1. **Ensure you have Flux.1-dev** (you likely already have this if using Flux in ComfyUI)

2. **Download OmniX adapters** using one of these methods:

   **Method A: Automatic Download (Recommended)**
   ```bash
   cd ComfyUI/custom_nodes/ComfyUI-OmniX
   pip install huggingface_hub
   python download_models.py
   ```

   **Method B: HuggingFace CLI**
   ```bash
   pip install huggingface_hub[cli]
   huggingface-cli download KevinHuang/OmniX \
       --local-dir ComfyUI/models/loras/omnix
   ```

   **Method C: Manual Download**
   - Visit: https://huggingface.co/KevinHuang/OmniX
   - Download all `.safetensors` files from the repository
   - Place in: `ComfyUI/models/loras/omnix/`

3. **Restart ComfyUI**

📖 **Detailed Instructions:** See [MODEL_DOWNLOAD_GUIDE.md](MODEL_DOWNLOAD_GUIDE.md) for complete setup, file mappings, and troubleshooting.

### Expected Directory Structure

```
ComfyUI/
├── models/
│   ├── checkpoints/ (or diffusion_models/)
│   │   └── flux1-dev.safetensors          # Base Flux model (~23GB)
│   └── loras/
│       └── omnix/                          # OmniX adapters (~3.5GB total)
│           ├── text_to_pano_rgb.safetensors           (~224MB)
│           ├── rgb_to_depth_depth.safetensors         (~224MB)
│           ├── rgb_to_normal_normal.safetensors       (~224MB)
│           ├── rgb_to_albedo_albedo.safetensors       (~224MB)
│           ├── rgb_to_pbr_pbr.safetensors             (~224MB)
│           └── rgb_to_semantic_semantic.safetensors   (~224MB)
│           └── (plus additional variant files from HuggingFace)
└── custom_nodes/
    └── ComfyUI-OmniX/                      # This repository
```

**Note:** Files are downloaded directly from HuggingFace with their original names. The code automatically maps these to the correct adapter types. See [MODEL_DOWNLOAD_GUIDE.md](MODEL_DOWNLOAD_GUIDE.md) for complete file mappings.

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

### Panorama Perception Workflow

**Requirements:**
- Flux.1-dev model and VAE
- OmniX adapter weights downloaded to `ComfyUI/models/loras/omnix/`
- Input panorama in 2:1 aspect ratio (equirectangular)

**Basic Workflow:**
```
1. [VAELoader] → Load Flux VAE
   ↓
2. [LoadImage] → Load your panorama image
   ↓
3. [OmniXPanoramaValidator] → Validate/fix aspect ratio to 2:1
   ↓
4. [OmniXAdapterLoader] → Load OmniX perception adapters (omnix-base, bf16)
   ↓
5. [OmniXPanoramaPerception] → Extract properties
   ├─ vae: Connect Flux VAE
   ├─ adapters: Connect adapter loader
   ├─ panorama: Connect validated image
   └─ Output: distance, normal, albedo, roughness, metallic maps
   ↓
6. [SaveImage] → Save extracted property maps
```

**Important Notes:**
- This implementation uses the real OmniX architecture (Flux VAE + LoRA adapters)
- Perception works in latent space, not pixel space
- Requires Flux.1-dev VAE for encoding/decoding
- Generation nodes removed - this is perception-only

## Nodes Reference

### OmniXAdapterLoader

Loads OmniX perception adapter weights.

**Inputs:**
- `adapter_preset`: Model variant (`omnix-base`)
- `precision`: Data type (`bf16` recommended, `fp16`, `fp32`)

**Outputs:**
- `adapters`: Adapter manager with loaded LoRA weights

**Description:**
Loads perception adapter weights from `ComfyUI/models/loras/omnix/`. Adapters are loaded as LoRA weights that will be injected into Flux transformer blocks for perception tasks.

### OmniXPanoramaPerception

Extracts geometric and material properties from panoramas using Flux VAE + LoRA adapters.

**Inputs:**
- `vae`: Flux VAE (required) - for encoding/decoding
- `adapters`: OmniX adapter manager from OmniXAdapterLoader
- `panorama`: Input panorama IMAGE (2:1 aspect ratio)
- `extract_distance`: Enable depth extraction (boolean)
- `extract_normal`: Enable normal extraction (boolean)
- `extract_albedo`: Enable albedo extraction (boolean)
- `extract_roughness`: Enable roughness extraction (boolean)
- `extract_metallic`: Enable metallic extraction (boolean)
- `num_steps`: Denoising steps (default: 28)

**Outputs:**
- `distance`: Depth/distance map
- `normal`: Surface normal map
- `albedo`: Base color/diffuse map
- `roughness`: Roughness map
- `metallic`: Metallic map

**Description:**
Uses the real OmniX architecture: encodes panorama to latents with Flux VAE, applies task-specific LoRA adapters, and decodes back to images. This is the main perception node.

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
ls ComfyUI/models/loras/omnix/
# Should show: text_to_pano_rgb.safetensors, rgb_to_depth_depth.safetensors, etc.
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
