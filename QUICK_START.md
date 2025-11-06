# ComfyUI-OmniX Quick Start Guide

**Get up and running with OmniX panorama generation in 5 minutes!**

## Step 1: Install ComfyUI-OmniX

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/cedarconnor/ComfyUI-OmniX.git
cd ComfyUI-OmniX
pip install -r requirements.txt
```

## Step 2: Download OmniX Adapters

### Option A: Automatic Download (Easiest)

```bash
# Download all adapters (~10GB)
python download_models.py

# Or download only what you need
python download_models.py --adapters rgb depth normal albedo

# List available adapters
python download_models.py --list
```

### Option B: HuggingFace CLI

```bash
pip install huggingface_hub[cli]

# Download all files
huggingface-cli download KevinHuang/OmniX \
    --local-dir ~/ComfyUI/models/omnix/omnix-base
```

## Step 3: Verify Installation

After download completes, check that files are in place:

```bash
ls -lh ~/ComfyUI/models/omnix/omnix-base/

# Should see:
# rgb_generation_adapter.safetensors (~2GB)
# distance_adapter.safetensors (~1.5GB)
# normal_adapter.safetensors (~1.5GB)
# albedo_adapter.safetensors (~1.5GB)
# pbr_adapter.safetensors (~2GB)
# semantic_adapter.safetensors (~1.5GB)
```

## Step 4: Restart ComfyUI

```bash
# Stop ComfyUI if running, then restart
cd ~/ComfyUI
python main.py
```

## Step 5: Create Your First Panorama

### Workflow 1: Text-to-Panorama

1. **Add nodes to canvas:**
   - `CheckpointLoaderSimple` → Load `flux1-dev.safetensors`
   - `OmniXAdapterLoader` → Select `omnix-base`, `fp16`
   - `OmniXApplyAdapters` → Connect MODEL and adapters, select `rgb_generation`
   - `CLIPTextEncode` → Write your prompt
   - `EmptyLatentImage` → Set to **2048×1024** (2:1 ratio)
   - `KSampler` → Steps: 28, CFG: 5.0
   - `VAEDecode` → Decode latents
   - `SaveImage` → Save result

2. **Example prompt:**
   ```
   "360 degree equirectangular view of a modern luxury penthouse living room,
   floor-to-ceiling windows overlooking city skyline, sunset lighting,
   photorealistic, architectural photography"
   ```

3. **Queue Prompt** and wait ~25 seconds

4. **Result:** A seamless 360° panorama!

### Workflow 2: Extract Properties from Panorama

1. **Add nodes:**
   - `LoadImage` → Load your panorama
   - `OmniXPanoramaValidator` → Fix aspect ratio if needed
   - `OmniXAdapterLoader` → Load perception adapters
   - `OmniXPanoramaPerception` → Enable all properties
   - `SaveImage` (×5) → Save each output

2. **Queue Prompt**

3. **Results:** Distance, normal, albedo, roughness, metallic maps!

## Pre-made Workflows

We include example workflows ready to use:

```bash
# Located in: ComfyUI-OmniX/workflows/

1. example_text_to_panorama.json
   - Complete text-to-360° generation
   - Just load and modify the prompt!

2. example_perception.json
   - Extract all properties from panorama
   - Outputs 5 different maps
```

**To use:**
1. In ComfyUI, click "Load" button
2. Navigate to `custom_nodes/ComfyUI-OmniX/workflows/`
3. Select a workflow
4. Modify parameters as needed
5. Queue!

## Available Adapters

| Adapter Name | Purpose | Use Case |
|-------------|---------|----------|
| **RGB Generation** | Text/image → panorama | Create 360° scenes |
| **Distance** | Depth prediction | 3D reconstruction |
| **Normal** | Surface normals | Relighting, bump maps |
| **Albedo** | Base color | Texture extraction |
| **PBR** | Roughness & metallic | Realistic materials |
| **Semantic** | Segmentation | Scene understanding |

## Tips for Best Results

### Panorama Generation

✅ **DO:**
- Use "360 degree", "equirectangular", or "panorama" in prompts
- Set 2:1 aspect ratio (2048×1024 or 1024×512)
- Use 28+ sampling steps for quality
- Try CFG scale 5.0-7.5

❌ **DON'T:**
- Use directional terms ("left side", "right corner")
- Set wrong aspect ratios
- Use too few steps (<20)

### Perception

✅ **DO:**
- Validate aspect ratio first (use OmniXPanoramaValidator)
- Start with fp16 precision for speed
- Disable unused properties to save VRAM

❌ **DON'T:**
- Feed non-panoramic images (aspect ratio matters!)
- Expect perfect results on low-quality inputs

## Performance Tips

**For 8GB VRAM:**
- Use fp16 precision
- Generate at 1024×512, upscale later
- Disable some perception properties
- Close other applications

**For 12GB VRAM:**
- Use fp16 precision
- Generate at 2048×1024 comfortably
- All features available

**For 16GB+ VRAM:**
- Use fp16 or fp32 precision
- Generate at 4096×2048 if desired
- Run perception on high-res panoramas

## Troubleshooting

### "Adapter weights not found"

```bash
# Re-run download
python download_models.py

# Or check installation
ls ~/ComfyUI/models/omnix/omnix-base/
```

### "Invalid panorama aspect ratio"

Solution: Add `OmniXPanoramaValidator` node before perception, set method to "crop"

### "Out of memory"

Solutions:
1. Lower resolution (1024×512)
2. Use fp16 precision
3. Disable some perception outputs
4. Restart ComfyUI to clear cache

### Generation looks wrong

Check:
- Using correct Flux.1-dev checkpoint
- Applied RGB generation adapter
- 2:1 aspect ratio in EmptyLatentImage
- Panorama-appropriate prompt

## Next Steps

- **Explore**: Try different prompts and settings
- **Experiment**: Combine generation + perception workflows
- **Share**: Post your creations (tag #ComfyUI #OmniX)
- **Contribute**: Open issues or PRs on GitHub

## Resources

- **Full Documentation**: [README.md](README.md)
- **Model Downloads**: [MODEL_DOWNLOAD_GUIDE.md](MODEL_DOWNLOAD_GUIDE.md)
- **Implementation Details**: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- **HuggingFace Model**: https://huggingface.co/KevinHuang/OmniX
- **Original OmniX**: https://github.com/HKU-MMLab/OmniX

---

**Happy panorama generating!** 🎨🌐

*Having issues? Open an issue on GitHub or check the troubleshooting section above.*
