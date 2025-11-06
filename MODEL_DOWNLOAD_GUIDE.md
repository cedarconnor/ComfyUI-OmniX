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

**Target Location:** `ComfyUI/models/omnix/omnix-base/`

---

## Download Methods

### Method 1: Automatic Download (Recommended)

We provide a Python script that automatically downloads and organizes all required files:

```bash
# From ComfyUI-OmniX directory
python download_models.py

# Or specify custom output directory
python download_models.py --output_dir /path/to/ComfyUI/models/omnix/omnix-base

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
    --local-dir ComfyUI/models/omnix/omnix-base \
    --local-dir-use-symlinks False
```

### Method 3: Manual Download from Web

1. Visit: https://huggingface.co/KevinHuang/OmniX
2. Click "Files and versions" tab
3. Download the following files:
   - `config.json`
   - All `.safetensors` files (adapters)
4. Place in: `ComfyUI/models/omnix/omnix-base/`

---

## Expected File Structure

After downloading, your directory should look like this:

```
ComfyUI/
├── models/
│   ├── checkpoints/
│   │   └── flux1-dev.safetensors          # Flux base model (if using checkpoints)
│   ├── omnix/
│   │   └── omnix-base/                     # OmniX adapters
│   │       ├── config.json                 # Configuration file
│   │       ├── rgb_adapter.safetensors     # For panorama generation (~2GB)
│   │       ├── distance_adapter.safetensors # For depth maps (~1.5GB)
│   │       ├── normal_adapter.safetensors  # For normals (~1.5GB)
│   │       ├── albedo_adapter.safetensors  # For albedo (~1.5GB)
│   │       ├── roughness_adapter.safetensors # For roughness (~1GB)
│   │       └── metallic_adapter.safetensors # For metallic (~1GB)
│   └── ...
└── custom_nodes/
    └── ComfyUI-OmniX/                      # This repository
```

---

## File Naming Reference

**IMPORTANT:** OmniX model files on HuggingFace may have different names than expected by our nodes. Here's the mapping:

| HuggingFace File | Rename To | Purpose | Size |
|-----------------|-----------|---------|------|
| *To be updated based on actual repo* | | | |

**If files need renaming**, use this command:
```bash
cd ComfyUI/models/omnix/omnix-base/

# Example renaming (update based on actual filenames):
# mv omnix_rgb.safetensors rgb_adapter.safetensors
# mv omnix_distance.safetensors distance_adapter.safetensors
# etc.
```

---

## Adapter Descriptions

### Generation Adapters

**RGB Generation Adapter** (`rgb_adapter.safetensors`)
- **Purpose**: Generates 360° panoramic images
- **Input**: Text prompt or image + text
- **Output**: Equirectangular panorama (2:1 aspect ratio)
- **Required for**: Text-to-panorama workflows

### Perception Adapters

**Distance Adapter** (`distance_adapter.safetensors`)
- **Purpose**: Predicts metric depth from panoramas
- **Output**: Distance/depth map
- **Use case**: 3D reconstruction, scene understanding

**Normal Adapter** (`normal_adapter.safetensors`)
- **Purpose**: Estimates surface normals
- **Output**: Normal map (XYZ vectors)
- **Use case**: Relighting, bump mapping, 3D reconstruction

**Albedo Adapter** (`albedo_adapter.safetensors`)
- **Purpose**: Extracts base color (diffuse texture)
- **Output**: Albedo map (RGB)
- **Use case**: PBR materials, texture extraction

**Roughness Adapter** (`roughness_adapter.safetensors`)
- **Purpose**: Estimates surface roughness
- **Output**: Roughness map (grayscale)
- **Use case**: PBR materials, realistic rendering

**Metallic Adapter** (`metallic_adapter.safetensors`)
- **Purpose**: Estimates metallic properties
- **Output**: Metallic map (grayscale)
- **Use case**: PBR materials, realistic rendering

---

## Verification

After installation, verify the setup:

```bash
# Check directory structure
ls -lh ComfyUI/models/omnix/omnix-base/

# Should show:
# config.json
# rgb_adapter.safetensors
# distance_adapter.safetensors
# normal_adapter.safetensors
# albedo_adapter.safetensors
# roughness_adapter.safetensors
# metallic_adapter.safetensors
```

**In ComfyUI:**
1. Restart ComfyUI
2. Add "OmniXAdapterLoader" node
3. Check if it loads without errors
4. Console should show: "✓ Loaded OmniX adapters: omnix-base (fp16)"

---

## Troubleshooting

### "Adapter directory not found"

**Solution:** Check the path
```bash
# Create directory if missing
mkdir -p ComfyUI/models/omnix/omnix-base/

# Verify ComfyUI can find it
# In ComfyUI, check: Settings → System → Folder Paths
```

### "Adapter weights not found: [filename]"

**Solution:** Verify file names match exactly
- Expected: `rgb_adapter.safetensors`
- Not: `rgb.safetensors` or `omnix_rgb.safetensors`
- Rename files to match expected names

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

If a larger variant becomes available:

```
ComfyUI/models/omnix/omnix-large/
├── config.json
├── rgb_adapter.safetensors
└── ... (larger adapter files)
```

Select in OmniXAdapterLoader node: `omnix-large`

---

## Storage Requirements

**Minimum (Generation only):**
- Flux.1-dev: 23GB
- RGB adapter: 2GB
- **Total: ~25GB**

**Recommended (All features):**
- Flux.1-dev: 23GB
- All OmniX adapters: 10GB
- **Total: ~33GB**

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
