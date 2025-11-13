PERCEPTION-ONLY | v1.1.0

# ComfyUI-OmniX

![OmniX Banner](https://img.shields.io/badge/ComfyUI-OmniX-blue) ![Version](https://img.shields.io/badge/version-1.1.0-green) ![License](https://img.shields.io/badge/license-Apache%202.0-orange)

**OmniX panorama perception for ComfyUI**

Extract comprehensive scene properties from 360-degree equirectangular panoramas, including depth, normals, and PBR materials, using OmniX adapters with Flux.

---

## Implementation Overview

ComfyUI-OmniX now focuses exclusively on the Diffusers-based pipeline. HuggingFace's `FluxPipeline` is loaded directly, OmniX LoRA adapters are attached using Diffusers' native APIs, and perception is run entirely within that pipeline for perfect compatibility with the original OmniX weights.

**Diffusers nodes provided:**

- `FluxDiffusersLoader` – loads Flux.1-dev from HuggingFace or a local checkpoint.
- `OmniXLoRALoader` – attaches the OmniX perception adapters to the pipeline.
- `OmniXPerceptionDiffusers` – encodes the panorama, runs conditioned denoising, and decodes the requested perception map.

See [DIFFUSERS_IMPLEMENTATION.md](DIFFUSERS_IMPLEMENTATION.md) for implementation details and diagnostics.

---

## Features

### Panorama Perception
- **Distance maps**: Extract metric depth from panoramas using Flux VAE + LoRA adapters
- **Normal maps**: Surface normals for downstream 3D reconstruction
- **Albedo maps**: Base color / diffuse extraction
- **PBR maps**: Roughness and metallic predictions from the same pass
- Multi-modal extraction that mirrors the reference OmniX architecture

### Architecture
- Flux VAE handles encoding/decoding entirely in latent space
- LoRA adapters are injected into Flux transformer blocks
- Matches the HKU-MMLab/OmniX implementation for behaviour parity

### Utility Nodes
- Panorama aspect ratio validation and correction
- Depth-map visualization helpers
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

**Detailed instructions:** See [MODEL_DOWNLOAD_GUIDE.md](MODEL_DOWNLOAD_GUIDE.md) for complete setup, file mappings, and troubleshooting.

### Expected Directory Structure

```
ComfyUI/
|-- models/
|   |-- checkpoints/ (or diffusion_models/)
|   |   `-- flux1-dev.safetensors
|   `-- loras/
|       `-- omnix/
|           |-- text_to_pano_rgb.safetensors
|           |-- rgb_to_depth_depth.safetensors
|           |-- rgb_to_normal_normal.safetensors
|           |-- rgb_to_albedo_albedo.safetensors
|           |-- rgb_to_pbr_pbr.safetensors
|           `-- rgb_to_semantic_semantic.safetensors
`-- custom_nodes/
    `-- ComfyUI-OmniX/
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
- Flux.1-dev checkpoint (local `.safetensors` or HuggingFace access)
- OmniX adapter weights downloaded to `ComfyUI/models/loras/omnix/`
- Input panorama in 2:1 aspect ratio (equirectangular)

**Basic Workflow:**

1. **LoadImage** – Load your panorama input (should already be 2:1). If not, preprocess it externally.
2. **FluxDiffusersLoader** – Choose `local_file` (pointing at your `flux1-dev.sft`) or `huggingface` plus the dtype.
3. **OmniXLoRALoader** – Point `adapter_dir` to `ComfyUI/models/loras/omnix/` and enable the adapters you need (distance/normal/albedo/pbr). This step loads the LoRAs once and keeps a patched pipeline alive for subsequent tasks.
4. **OmniXPerceptionDiffusers** – Connect the patched pipeline, the `loaded_adapters`, and your panorama image. Select the task (`distance`, `normal`, `albedo`, `pbr`), the number of inference steps, guidance scale, and noise strength. The output is a single perception map.
5. **SaveImage** – Save the resulting tensor. Duplicate step 4/5 for each perception head you want to export.

**Notes:**
- `noise_strength` controls how far the denoiser drifts from the encoded panorama (lower values keep more of the source image).
- The PBR adapter generates roughness/metallic outputs; run the node twice with `task="pbr"` if you want both maps.
- Because everything lives inside the Diffusers pipeline, there is no need for ComfyUI MODEL/CLIP/VAE inputs anymore.

## Nodes Reference

### FluxDiffusersLoader

Loads the HuggingFace `FluxPipeline` either from a local `.safetensors` file (`load_method="local_file"`) or directly from the Hub (`"huggingface"`). It exposes the pipeline plus its VAE/text encoder for downstream use.

### OmniXLoRALoader

Attaches OmniX LoRA adapters (distance, normal, albedo, PBR) to the loaded Diffusers pipeline. You can toggle which adapters are enabled and set a global scale. Returns the updated pipeline and a metadata dict describing which adapters were applied.

### OmniXPerceptionDiffusers

Runs a single perception task. Inputs:

- `flux_pipeline`: The patched pipeline from `OmniXLoRALoader`.
- `loaded_adapters`: Metadata dict (used for validation/logging).
- `panorama`: Panorama image tensor in ComfyUI format.
- `task`: `"distance"`, `"normal"`, `"albedo"`, or `"pbr"`.
- `num_steps`: Denoising steps (default 28).
- `guidance_scale`: CFG scale (default 3.5).
- `noise_strength`: How strongly to perturb the encoded panorama before denoising.

Output: `perception_output` (the requested map). Duplicate the node with different `task` values to export multiple maps in one workflow.

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
| Perception (single head) | 2048x1024 | ~8s | 28 steps, bf16 |

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
2. Reduce output resolution (e.g., 1024x512 instead of 2048x1024)
3. Close other applications using VRAM
4. Enable model offloading in Flux checkpoint loader
5. Disable some perception modes (extract only what you need)

### "Invalid panorama aspect ratio"

**Solution:** Ensure the source panorama is 2:1 before running perception. Crop or pad it externally (e.g., in an image editor or via another ComfyUI node) so the width is exactly double the height.

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
- **RGB Generation**: Guides Flux to generate 360 degrees panoramas
- **Distance**: Predicts metric depth values
- **Normal**: Estimates surface normals
- **Albedo/Roughness/Metallic**: Extracts PBR material properties

Adapters are lightweight (1-2GB each) compared to the base model (23GB) and use cross-modal learning for high-quality outputs.

### Format Specifications

**Panorama Format:**
- Type: Equirectangular projection
- Aspect Ratio: 2:1 (width:height)
- Supported Resolutions: 512x1024, 1024x2048, 2048x4096
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

**Made with love for the ComfyUI community**
