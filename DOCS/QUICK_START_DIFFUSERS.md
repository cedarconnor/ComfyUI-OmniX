# Quick Start: Diffusers Approach

**Get OmniX perception working in 5 minutes!**

---

## Prerequisites ✅

Already done:
- ✅ ComfyUI installed
- ✅ Diffusers library installed (`diffusers>=0.29.0`)
- ✅ OmniX adapters downloaded to `C:/ComfyUI/models/loras/omnix/`

---

## Step 1: Restart ComfyUI

```bash
# Close ComfyUI if running
# Then start it again to load the new nodes
```

This reloads the Python modules including our new Diffusers nodes.

**Note**: If using a local Flux model file (`.sft` or `.ckpt`), the loader will automatically handle PyTorch 2.6+ security policies. You may see a message about "alternative loading method" - this is normal and safe for trusted local files.

---

## Step 2: Create Workflow

### Option A: Manual Setup

1. **Add FluxDiffusersLoader node**
   - Find in: `OmniX/Diffusers` category
   - Settings:
     - `load_method`: "local_file" (if you have Flux downloaded) OR "huggingface" (to download)
     - `local_checkpoint`: Select your local file (e.g., "flux1-dev.sft") if using local_file
     - `model_id`: "black-forest-labs/FLUX.1-dev" (if using huggingface method)
     - `torch_dtype`: "bfloat16" (recommended for best speed/quality)

2. **Add OmniXLoRALoader node**
   - Connect `flux_pipeline` from FluxDiffusersLoader
   - Settings:
     - `adapter_dir`: "C:/ComfyUI/models/loras/omnix"
     - Enable: `distance`, `normal`, `albedo`
     - `lora_scale`: 1.0

3. **Add LoadImage node**
   - Load your panorama (1024×2048 recommended)

4. **Add OmniXPerceptionDiffusers node**
   - Connect:
     - `flux_pipeline` from OmniXLoRALoader
     - `loaded_adapters` from OmniXLoRALoader
     - `panorama` from LoadImage
   - Settings:
     - `task`: "distance"
     - `num_steps`: 5 (for quick test)
     - `guidance_scale`: 3.5
     - `noise_strength`: 0.1

5. **Add SaveImage node**
   - Connect `perception_output` from OmniXPerceptionDiffusers
   - Set prefix: "distance_test"

---

## Step 3: Run Test

1. **Queue Prompt** (or press Ctrl+Enter)

2. **First Run**: Will download FLUX.1-dev (~30GB)
   - This happens once
   - Cached for future use
   - Takes 10-30 minutes depending on internet

3. **Expected Console Output**:
```
[FluxDiffusers] Loading Flux from black-forest-labs/FLUX.1-dev
[FluxDiffusers] Device: cuda:0, dtype: torch.bfloat16
[FluxDiffusers] ✓ Pipeline loaded successfully

[OmniXLoRA] Loading adapters: ['distance', 'normal', 'albedo']
[OmniXLoRA] Loading distance from C:\ComfyUI\models\loras\omnix\rgb_to_depth_depth.safetensors
[OmniXLoRA] ✓ Loaded distance: 456 weights
[OmniXLoRA] Loading normal from C:\ComfyUI\models\loras\omnix\rgb_to_normal_normal.safetensors
[OmniXLoRA] ✓ Loaded normal: 456 weights
[OmniXLoRA] Loading albedo from C:\ComfyUI\models\loras\omnix\rgb_to_albedo_albedo.safetensors
[OmniXLoRA] ✓ Loaded albedo: 456 weights
[OmniXLoRA] ✓ Loaded 3 adapters

[OmniXPerception] Running distance perception
[OmniXPerception] Steps: 5, Guidance: 3.5, Noise: 0.1
[OmniXPerception] Input tensor shape: torch.Size([1, 3, 1024, 2048])
[OmniXPerception] Active adapter: distance
[OmniXPerception] Encoded to latents: torch.Size([1, 16, 128, 256])
[OmniXPerception] ✓ Inference complete
```

4. **Check Output**:
   - Look in ComfyUI output folder
   - File: `distance_test_00001_.png`
   - Should show: Depth-like gradient (not just a copy of input)

---

## Step 4: Run All Tasks

Duplicate the `OmniXPerceptionDiffusers` node twice:
- Node 1: `task` = "distance"
- Node 2: `task` = "normal"
- Node 3: `task` = "albedo"

Connect each to separate SaveImage nodes.

Run workflow - you'll get 3 different outputs!

---

## Step 5: Increase Quality

Once confirmed working:
- Change `num_steps`: 5 → 28 (OmniX default)
- Wait ~60s per task
- Much better quality

---

## Troubleshooting

### "Model not found" or Download Fails
**Issue**: Can't download FLUX.1-dev
**Solutions**:
- Check internet connection
- Ensure ~35GB free disk space
- Try again (downloads resume)
- Or download manually:
  ```bash
  huggingface-cli download black-forest-labs/FLUX.1-dev
  ```

### "Weights only load failed" Error
**Issue**: PyTorch 2.6+ security policy preventing loading of `.sft` or `.ckpt` files
**Solution**: The loader automatically handles this! Look for console message:
```
[FluxDiffusers] Standard loading failed due to PyTorch 2.6+ security policy
[FluxDiffusers] Attempting alternative loading method...
[FluxDiffusers] ✓ Loaded using alternative method (weights_only=False)
```
This is safe for your trusted local Flux model file. If it still fails, ensure you're using the latest version of the nodes.

### "Adapter not found"
**Issue**: LoRA files missing
**Solution**: Verify files exist:
```
C:\ComfyUI\models\loras\omnix\
├── rgb_to_depth_depth.safetensors
├── rgb_to_normal_normal.safetensors
├── rgb_to_albedo_albedo.safetensors
└── rgb_to_pbr_pbr.safetensors
```

### Out of Memory (OOM)
**Issue**: GPU VRAM exhausted
**Solutions**:
1. Use smaller image (512×1024)
2. Reduce `num_steps` to 3-5
3. Close other GPU applications

### Slow Inference
**Issue**: Takes too long
**Solutions**:
- Already using bfloat16 (fastest)
- Reduce `num_steps`
- Ensure CUDA is working (check device in console)

### Output Looks Wrong
**Issue**: Not getting perception maps
**Debug**:
1. Check console for errors
2. Verify adapter loaded: "✓ Loaded distance: 456 weights"
3. Try different `lora_scale` (0.5, 1.0, 1.5)
4. Increase `num_steps` to 28
5. Share console output for help

---

## Expected Results

### Distance Map
- Grayscale depth representation
- Near objects: Darker/brighter (depends on encoding)
- Far objects: Opposite
- Clear depth gradation

### Normal Map
- Colorful RGB image
- R: X surface direction
- G: Y surface direction
- B: Z surface direction
- Looks like colored 3D surface

### Albedo Map
- Base material colors
- Flatter lighting than input
- Shows intrinsic color without shading

**All three should look distinctly different from each other and from the input!**

---

## Performance

**RTX 4090** (estimated):
- 5 steps: ~10s per task
- 28 steps: ~60s per task

**RTX 3090** (estimated):
- 5 steps: ~15s per task
- 28 steps: ~90s per task

**Slower GPUs**:
- Use 5 steps for testing
- Consider 512×1024 resolution

---

## Next Steps

### If Working ✅
1. Test all panoramas in your collection
2. Experiment with parameters:
   - `num_steps`: 10, 20, 28, 50
   - `guidance_scale`: 1.0, 3.5, 7.0
   - `noise_strength`: 0.05, 0.1, 0.15, 0.2
   - `lora_scale`: 0.5, 1.0, 1.5
3. Compare quality with OmniX paper results
4. Share your results!

### If Not Working ⚠️
1. Share full console output
2. Share workflow JSON
3. Check VRAM usage
4. Try simpler test (512×1024, 3 steps)
5. Open issue on GitHub

---

## Full Documentation

- **Implementation Details**: [DIFFUSERS_IMPLEMENTATION.md](../DIFFUSERS_IMPLEMENTATION.md)
- **Session Journey**: [SESSION_SUMMARY.md](SESSION_SUMMARY.md)
- **Main README**: [README.md](../README.md)

---

## Quick Reference

### Recommended Settings

**Fast Testing**:
- `num_steps`: 5
- `guidance_scale`: 3.5
- `noise_strength`: 0.1

**High Quality**:
- `num_steps`: 28
- `guidance_scale`: 3.5
- `noise_strength`: 0.1

**Maximum Quality**:
- `num_steps`: 50
- `guidance_scale`: 5.0
- `noise_strength`: 0.15

### Node Connections

```
LoadImage → OmniXPerceptionDiffusers → SaveImage
             ↑                    ↑
FluxDiffusersLoader → OmniXLoRALoader
```

---

**Ready to test! Good luck!** 🚀

If it works, you'll have OmniX perception running in ComfyUI!
If not, we'll debug together.
