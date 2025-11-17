# ComfyUI-OmniX

<div align="center">

**Panoramic Perception for ComfyUI**

Integrate [OmniX](https://github.com/HKU-MMLab/OmniX) panoramic generation and perception capabilities into ComfyUI using FLUX.1-dev.

[Installation](#installation) • [Quick Start](#quick-start) • [Workflows](#workflows) • [Documentation](#documentation)

</div>

---

## What is OmniX?

**OmniX** is a powerful panoramic generation and perception system built on FLUX.1-dev that enables:

- **Text-to-Panorama Generation**: Create 360° RGB panoramas from text prompts
- **Depth Estimation**: Extract metric depth maps from panoramas
- **Surface Normal Estimation**: Compute geometric surface normals
- **PBR Material Estimation**: Generate physically-based rendering materials (albedo, roughness, metallic)
- **Semantic Segmentation**: Identify and segment scene elements

This ComfyUI custom node pack brings these capabilities to your ComfyUI workflows.

---

## Features

- ✨ **Native ComfyUI Integration**: Works seamlessly with existing ComfyUI nodes
- 🎨 **Multiple Perception Modes**: Depth, normals, PBR, and semantic segmentation
- 🔧 **Flexible Post-Processing**: Customizable normalization, colorization, and visualization
- 📦 **Easy Installation**: Simple setup process
- 🚀 **Optimized Performance**: Efficient VAE encoding/decoding with FLUX

---

## Installation

### Step 1: Install ComfyUI-OmniX

Navigate to your ComfyUI custom nodes directory and clone this repository:

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/cedarconnor/ComfyUI-OmniX.git
cd ComfyUI-OmniX
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Most dependencies (torch, numpy, PIL) should already be installed with ComfyUI. This step ensures you have the additional packages needed for perception processing.

### Step 3: Download FLUX.1-dev Base Model

Download the FLUX.1-dev checkpoint and place it in:
```
ComfyUI/models/checkpoints/flux1-dev.safetensors
```

You can get it from:
- HuggingFace: https://huggingface.co/black-forest-labs/FLUX.1-dev
- Or use the model you already have

### Step 4: Download and Convert OmniX LoRAs

#### Option A: Download from HuggingFace

```bash
# Clone the OmniX model repository
cd ~/models  # or your preferred location
git lfs install
git clone https://huggingface.co/KevinHuang/OmniX
```

#### Option B: Direct Download

Download individual LoRA files from: https://huggingface.co/KevinHuang/OmniX/tree/main

You need these files:
- `lora_rgb_to_albedo/` (or `lora_rgb_to_albedo.safetensors`)
- `lora_rgb_to_depth/`
- `lora_rgb_to_normal/`
- `lora_rgb_to_pbr/`
- `lora_rgb_to_semantic/`
- `lora_text_to_pano/` (for text → panorama generation)

#### Convert LoRAs to ComfyUI Format

```bash
cd ComfyUI/custom_nodes/ComfyUI-OmniX

python scripts/convert_loras.py \
    --input ~/models/OmniX \
    --output ~/ComfyUI/models/loras
```

This will create files like:
- `OmniX_text_to_pano_comfyui.safetensors`
- `OmniX_rgb_to_depth_comfyui.safetensors`
- `OmniX_rgb_to_normal_comfyui.safetensors`
- `OmniX_rgb_to_albedo_comfyui.safetensors`
- `OmniX_rgb_to_pbr_comfyui.safetensors`
- `OmniX_rgb_to_semantic_comfyui.safetensors`

### Step 5: Restart ComfyUI

Restart ComfyUI to load the new custom nodes. You should see **OmniX/Perception** category with four nodes:
- OmniX Pano Perception: Depth
- OmniX Pano Perception: Normal
- OmniX Pano Perception: PBR
- OmniX Pano Perception: Semantic

---

## Quick Start

### Generate a Panorama from Text

1. **Load Checkpoint** → `flux1-dev.safetensors`
2. **Load LoRA** → `OmniX_text_to_pano_comfyui.safetensors` (strength: 1.0)
3. **CLIP Text Encode** → `"a photorealistic 360° panorama of a cozy modern living room, equirectangular, 8k"`
4. **Empty Latent Image** → Width: 2048, Height: 1024
5. **KSampler** → Steps: 28, CFG: 3.5
6. **VAE Decode** → Preview your panorama!

### Extract Depth from Panorama

1. **Load Image** → Your panorama
2. **VAE Encode** → Encode to latent
3. **Load Checkpoint** → `flux1-dev.safetensors`
4. **Load LoRA** → `OmniX_rgb_to_depth_comfyui.safetensors`
5. **KSampler** → With panorama latent as conditioning
6. **OmniX Pano Perception: Depth** → Process and visualize depth
   - Enable colorize for better visualization
7. **Preview Image** → See your depth map!

For detailed workflow guides, see [examples/WORKFLOWS.md](examples/WORKFLOWS.md)

---

## Node Reference

### OmniX Pano Perception: Depth

Extracts and visualizes depth maps from panoramic images.

**Inputs:**
- `vae` (VAE): VAE model for decoding latents
- `samples` (LATENT): Output from KSampler with depth LoRA
- `normalize` (BOOLEAN): Normalize depth to [0, 1] range (default: True)
- `colorize` (BOOLEAN): Apply colormap for visualization (default: False)
- `colormap` (ENUM): Colormap to use - `inferno`, `viridis`, `plasma`, `turbo`, `gray`

**Outputs:**
- `depth_img` (IMAGE): Depth visualization

### OmniX Pano Perception: Normal

Extracts and visualizes surface normals from panoramic images.

**Inputs:**
- `vae` (VAE): VAE model for decoding
- `samples` (LATENT): Output from KSampler with normal LoRA
- `normalize_vectors` (BOOLEAN): Normalize normal vectors to unit length (default: True)
- `output_format` (ENUM): `rgb` or `normalized`

**Outputs:**
- `normal_img` (IMAGE): Normal map visualization (RGB encoding of XYZ normals)

### OmniX Pano Perception: PBR

Extracts PBR material properties from panoramic images.

**Inputs:**
- `vae` (VAE): VAE model for decoding
- `samples` (LATENT): Output from KSampler with PBR LoRA
- `normalize_maps` (BOOLEAN): Normalize each map to [0, 1] (default: True)

**Outputs:**
- `albedo_img` (IMAGE): Albedo/diffuse color map
- `roughness_img` (IMAGE): Surface roughness map
- `metallic_img` (IMAGE): Metallic property map

### OmniX Pano Perception: Semantic

Extracts semantic segmentation from panoramic images.

**Inputs:**
- `vae` (VAE): VAE model for decoding
- `samples` (LATENT): Output from KSampler with semantic LoRA
- `palette` (ENUM): Color palette - `ade20k`, `cityscapes`, `custom`
- `num_classes` (INT): Number of semantic classes (default: 16)

**Outputs:**
- `semantic_img` (IMAGE): Colorized semantic segmentation map

---

## Workflows

Three main workflow patterns are supported:

### W1: Text → Panorama
Generate 360° panoramas from text descriptions.

### W2: Panorama → Perception
Extract depth, normals, PBR, or semantics from existing panoramas.

### W3: Text → Panorama → Perception
Full pipeline from text prompt to perception outputs.

See detailed workflow guides in [examples/WORKFLOWS.md](examples/WORKFLOWS.md)

---

## Documentation

- **[Design Document](design_doc.md)**: Technical architecture and implementation details
- **[Agent Roles](agents.md)**: Development workflow and maintenance roles
- **[Workflow Examples](examples/WORKFLOWS.md)**: Detailed workflow building guides
- **[Script Documentation](scripts/README.md)**: LoRA conversion and utilities

---

## Tips & Best Practices

### Prompting for Panoramas

- Include keywords: `360°`, `panoramic`, `equirectangular`
- Specify scene type: `interior`, `outdoor`, `studio`
- Add quality modifiers: `8k`, `highly detailed`, `photorealistic`
- Use negative prompts: `blurry, distorted, low quality`

### Resolution Recommendations

- **Standard**: 1024 × 2048 (good balance of quality and speed)
- **High Quality**: 1536 × 3072 (requires more VRAM)
- **Maximum**: 2048 × 4096 (very high VRAM usage)

Always maintain 1:2 aspect ratio for proper equirectangular format.

### Performance Optimization

- **Enable CPU offload** if running low on VRAM
- **Process perception tasks sequentially** rather than parallel if memory-constrained
- **Use lower resolution** for testing, then upscale final outputs
- **Clear VRAM** between different perception types

### Output Quality

- **Depth**: Enable colorization for easier interpretation
- **Normals**: Use `normalize_vectors` for physically accurate results
- **PBR**: Results work best with well-lit panoramas
- **Semantic**: Custom palette provides most distinct class colors

---

## Troubleshooting

### Nodes Don't Appear in ComfyUI

1. Check that `__init__.py` exists in the node directory
2. Look for import errors in ComfyUI console
3. Ensure all dependencies are installed
4. Restart ComfyUI completely

### LoRAs Not Found

1. Verify LoRAs are in `ComfyUI/models/loras/`
2. Check filenames match exactly (case-sensitive)
3. Try refreshing the node list (F5 in browser)
4. Restart ComfyUI

### Out of Memory Errors

1. Reduce panorama resolution
2. Enable model CPU offloading
3. Process one perception type at a time
4. Close other GPU applications

### Poor Quality Outputs

1. Increase KSampler steps (try 40-50)
2. Adjust CFG scale (3.0-5.0 range)
3. Ensure correct LoRA is loaded for each task
4. Try different seeds
5. Check input panorama quality

### Seams in Panoramas

Equirectangular images naturally have a seam where left meets right. OmniX includes blending to minimize this, but some seams may be visible. Post-process in external software if needed.

---

## Architecture

ComfyUI-OmniX follows a clean, modular architecture:

```
ComfyUI-OmniX/
├── __init__.py              # Node registration
├── omni_x_nodes.py          # Node implementations
├── omni_x_utils.py          # Core perception utilities
├── requirements.txt         # Dependencies
├── design_doc.md            # Technical specification
├── agents.md                # Development workflow
├── scripts/
│   ├── convert_loras.py     # LoRA conversion tool
│   └── README.md            # Script documentation
└── examples/
    └── WORKFLOWS.md         # Workflow guides
```

### Key Design Principles

1. **Separation of Concerns**: Nodes handle I/O, utilities handle processing
2. **ComfyUI Native**: Works with standard ComfyUI node patterns
3. **Modular**: Each perception type is independent
4. **Extensible**: Easy to add new perception modes

---

## Limitations

- **FLUX.1-dev Only**: Currently only supports FLUX.1-dev base model
- **Equirectangular Format**: Designed for 1:2 aspect ratio panoramas
- **LoRA Required**: Each perception type requires its specific LoRA
- **GPU Memory**: High-resolution panoramas require significant VRAM

---

## Credits

- **OmniX**: Original research and models by HKU-MMLab
  - Paper: [OmniX: Omni-Directional Panoramic Generation and Perception](https://github.com/HKU-MMLab/OmniX)
  - Models: https://huggingface.co/KevinHuang/OmniX
- **FLUX.1**: Base diffusion model by Black Forest Labs
- **ComfyUI**: Node-based interface by comfyanonymous

---

## License

This node pack is a community integration layer. Please respect:
- OmniX original license and usage terms
- FLUX.1-dev license and restrictions
- ComfyUI license

See individual component repositories for specific license details.

---

## Contributing

Contributions are welcome! Areas for improvement:

- Additional perception modes (relighting, 3D reconstruction)
- Workflow templates and examples
- Performance optimizations
- Documentation improvements

---

## Support

- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Share workflows and tips in GitHub Discussions
- **OmniX**: For questions about the underlying models, see the [OmniX repository](https://github.com/HKU-MMLab/OmniX)

---

## Changelog

### v0.1.0 (Initial Release)

- ✨ Initial implementation of four perception nodes
- 📦 LoRA conversion script
- 📚 Complete documentation and workflow guides
- 🎨 Support for depth, normal, PBR, and semantic perception

---

<div align="center">

**Made with ❤️ for the ComfyUI community**

[Report Bug](https://github.com/cedarconnor/ComfyUI-OmniX/issues) · [Request Feature](https://github.com/cedarconnor/ComfyUI-OmniX/issues) · [Documentation](examples/WORKFLOWS.md)

</div>
