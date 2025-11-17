# OmniX Scripts

This directory contains utility scripts for working with OmniX in ComfyUI.

## convert_loras.py

Converts OmniX LoRAs from HuggingFace Diffusers format to ComfyUI format.

### Prerequisites

```bash
pip install safetensors torch
```

### Usage

If you downloaded the LoRAs from HuggingFace (https://huggingface.co/KevinHuang/OmniX):

```bash
python scripts/convert_loras.py \
    --input /path/to/downloaded/omnix/loras \
    --output /path/to/ComfyUI/models/loras
```

### Example

```bash
# If you cloned the HuggingFace repo
python scripts/convert_loras.py \
    --input ~/models/OmniX \
    --output ~/ComfyUI/models/loras

# If you only want specific tasks
python scripts/convert_loras.py \
    --input ~/models/OmniX \
    --output ~/ComfyUI/models/loras \
    --tasks rgb_to_depth rgb_to_normal
```

### Output

The script will create files like:
- `OmniX_text_to_pano_comfyui.safetensors`
- `OmniX_rgb_to_depth_comfyui.safetensors`
- `OmniX_rgb_to_normal_comfyui.safetensors`
- `OmniX_rgb_to_albedo_comfyui.safetensors`
- `OmniX_rgb_to_pbr_comfyui.safetensors`
- `OmniX_rgb_to_semantic_comfyui.safetensors`

### Troubleshooting

**"LoRA file not found"**
- The script searches for LoRAs in various naming patterns
- Check that your input directory contains the OmniX LoRA files
- The LoRAs may be in subdirectories named after each task

**"Verification failed"**
- The LoRA file may be corrupted or in an unexpected format
- Try re-downloading from HuggingFace
