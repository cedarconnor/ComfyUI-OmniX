# OmniX Model Download Guide

This guide explains how to download and install OmniX adapter weights for ComfyUI-OmniX.

## Overview

OmniX consists of two main components:
1. **Base Model**: Flux.1-dev (you likely already have this for ComfyUI)
2. **OmniX Adapters**: Specialized adapters for panorama generation and perception

## Required Models

### 1. Flux.1-dev Base Model

**Already have it?** If you're using Flux in ComfyUI, you already have this!

**Location:** `ComfyUI/models/checkpoints/` or `ComfyUI/models/diffusion_models/`

**If you need to download it:**
- Size: ~23GB
- Source: https://huggingface.co/black-forest-labs/FLUX.1-dev
- Download via ComfyUI Manager or manually

### 2. OmniX Adapters

**Source:** https://huggingface.co/KevinHuang/OmniX

**Total Size:** ~10-12GB (all adapters)

**Target Location:** `ComfyUI/models/loras/omnix/`

---

## Download Methods

### Method 1: Automatic Download (Recommended)

We provide a Python script that automatically downloads and organizes all required files:

```bash
# From ComfyUI-OmniX directory
python download_models.py

# Or specify custom output directory
python download_models.py --output_dir /path/to/ComfyUI/models/loras/omnix

# List available files first
python download_models.py --list-only
```

**Prerequisites:**
```bash
pip install huggingface_hub
```

### Method 2: HuggingFace CLI

```bash
# Install HuggingFace CLI
pip install huggingface_hub[cli]

# Download all files
huggingface-cli download KevinHuang/OmniX \
    --local-dir ComfyUI/models/loras/omnix \
    --local-dir-use-symlinks False
```

### Method 3: Manual Download from Web

1. Visit: https://huggingface.co/KevinHuang/OmniX
2. Click "Files and versions" tab
3. Download all `.safetensors` files (adapters)
   - Keep their original filenames (e.g., `text_to_pano_rgb.safetensors`)
4. Place in: `ComfyUI/models/loras/omnix/`

---

## Expected File Structure

After downloading, your directory should look like this:

```
ComfyUI/
├── models/
│   ├── checkpoints/ (or diffusion_models/)
│   │   └── flux1-dev.safetensors          # Flux base model (~23GB)
│   ├── loras/
│   │   └── omnix/                          # OmniX adapters
│   │       ├── text_to_pano_rgb.safetensors           (~224MB)
│   │       ├── rgb_to_depth_depth.safetensors         (~224MB)
│   │       ├── rgb_to_normal_normal.safetensors       (~224MB)
│   │       ├── rgb_to_albedo_albedo.safetensors       (~224MB)
│   │       ├── rgb_to_pbr_pbr.safetensors             (~224MB)
│   │       └── rgb_to_semantic_semantic.safetensors   (~224MB)
│   │       └── (plus additional variant files)
│   └── ...
└── custom_nodes/
    └── ComfyUI-OmniX/                      # This repository
```

---

## File Naming Reference

**IMPORTANT:** ComfyUI-OmniX now uses the **original HuggingFace filenames** directly. No renaming is required!

### HuggingFace Files to Adapter Type Mapping

The code automatically maps HuggingFace filenames to adapter types:

| HuggingFace Filename | Adapter Type | Purpose | Size |
|---------------------|--------------|---------|------|
| `text_to_pano_rgb.safetensors` | `rgb_generation` | Text/image → panorama | ~224MB |
| `rgb_to_depth_depth.safetensors` | `distance` | Depth prediction | ~224MB |
| `rgb_to_normal_normal.safetensors` | `normal` | Normal mapping | ~224MB |
| `rgb_to_albedo_albedo.safetensors` | `albedo` | Albedo extraction | ~224MB |
| `rgb_to_pbr_pbr.safetensors` | `pbr` | Roughness & metallic | ~224MB |
| `rgb_to_semantic_semantic.safetensors` | `semantic` | Semantic segmentation | ~224MB |

**Additional files in the repository** (e.g., `*_rgb.safetensors`, `*_camray.safetensors`) are variants and intermediate outputs that are not currently used by the ComfyUI integration. You can download them for completeness, but they are optional.

### Manual Download Instructions

Simply download the files from https://huggingface.co/KevinHuang/OmniX and keep their original names:

```bash
# No renaming needed! Just place files in the correct directory:
# ComfyUI/models/loras/omnix/text_to_pano_rgb.safetensors
# ComfyUI/models/loras/omnix/rgb_to_depth_depth.safetensors
# etc.
```

---

## Adapter Descriptions

### Generation Adapters

**RGB Generation Adapter** (`text_to_pano_rgb.safetensors`)
- **Purpose**: Generates 360° panoramic images
- **Input**: Text prompt or image + text
- **Output**: Equirectangular panorama (2:1 aspect ratio)
- **Required for**: Text-to-panorama workflows

### Perception Adapters

**Distance Adapter** (`rgb_to_depth_depth.safetensors`)
- **Purpose**: Predicts metric depth from panoramas
- **Output**: Distance/depth map
- **Use case**: 3D reconstruction, scene understanding

**Normal Adapter** (`rgb_to_normal_normal.safetensors`)
- **Purpose**: Estimates surface normals
- **Output**: Normal map (XYZ vectors)
- **Use case**: Relighting, bump mapping, 3D reconstruction

**Albedo Adapter** (`rgb_to_albedo_albedo.safetensors`)
- **Purpose**: Extracts base color (diffuse texture)
- **Output**: Albedo map (RGB)
- **Use case**: PBR materials, texture extraction

**PBR Adapter** (`rgb_to_pbr_pbr.safetensors`)
- **Purpose**: Estimates surface roughness and metallic properties
- **Output**: Combined roughness + metallic maps
- **Use case**: PBR materials, realistic rendering
- **Note**: This single adapter handles both roughness and metallic extraction

---

## Verification

After installation, verify the setup:

```bash
# Check directory structure
ls -lh ComfyUI/models/loras/omnix/

# Should show:
# text_to_pano_rgb.safetensors
# rgb_to_depth_depth.safetensors
# rgb_to_normal_normal.safetensors
# rgb_to_albedo_albedo.safetensors
# rgb_to_pbr_pbr.safetensors
# rgb_to_semantic_semantic.safetensors
# (plus additional variant files)
```

**In ComfyUI:**
1. Restart ComfyUI
2. Add "OmniXAdapterLoader" node
3. Check if it loads without errors
4. Console should show: "✓ Loaded OmniX adapters: omnix-base (fp16)" and the adapter path

---

## Troubleshooting

### "Adapter directory not found"

**Solution:** Check the path
```bash
# Create directory if missing
mkdir -p ComfyUI/models/loras/omnix/

# Verify ComfyUI can find it
# In ComfyUI, check: Settings → System → Folder Paths
```

### "Adapter weights not found: [filename]"

**Solution:** Verify file names match the HuggingFace originals
- Expected: `text_to_pano_rgb.safetensors`, `rgb_to_depth_depth.safetensors`, etc.
- Keep the original HuggingFace filenames - no renaming needed!
- Make sure files are in `ComfyUI/models/loras/omnix/`

### "403 Forbidden" when downloading

**Solution:** The repository may be gated
1. Create HuggingFace account: https://huggingface.co/join
2. Accept model terms if required
3. Login via CLI:
   ```bash
   huggingface-cli login
   ```
4. Retry download

### Files are too large / slow download

**Solution:** Download individually
```bash
# Download only what you need
python download_models.py --adapters rgb distance normal

# Or use HuggingFace CLI for specific files
huggingface-cli download KevinHuang/OmniX rgb_adapter.safetensors
```

---

## Optional: Omnix-Large Variant

If a larger variant becomes available, you could organize it in a separate subdirectory:

```
ComfyUI/models/loras/
├── omnix/              # Base variant (current)
│   └── *.safetensors
└── omnix-large/        # Large variant (if available)
    └── *.safetensors
```

**Note:** Currently only the base variant is available from the HuggingFace repository.

---

## Storage Requirements

**Minimum (Generation only):**
- Flux.1-dev: 23GB
- RGB adapter: 224MB
- **Total: ~23.2GB**

**Recommended (All features):**
- Flux.1-dev: 23GB
- All core OmniX adapters: ~1.5GB (6 files × 224MB each)
- All files from HuggingFace: ~3.5GB (includes variants)
- **Total: ~26.5GB**

**Plan ahead:** Ensure you have sufficient disk space before downloading.

---

## Alternative Sources

If HuggingFace is unavailable:

1. **Official GitHub**: https://github.com/HKU-MMLab/OmniX
   - Check releases page for direct downloads

2. **Mirror repositories**: Check if community mirrors exist

3. **Contact authors**: For research purposes, contact the OmniX team

---

## Updates

This guide will be updated as:
- New adapter variants are released
- File naming conventions change
- Additional download methods become available

**Last updated:** November 2025

**Questions?** Open an issue: https://github.com/[YOUR_USERNAME]/ComfyUI-OmniX/issues
