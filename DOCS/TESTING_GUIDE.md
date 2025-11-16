# Testing Guide for OmniX Perception Phase 1

> **Legacy note:** This document describes the removed ComfyUI-native perception nodes. The active Diffusers-based workflow is documented in `README.md`. Keep this guide only as a historical reference for Phase 1 testing.

**Status**: Phase 1 implementation complete - Ready for testing
**Date**: January 12, 2025

---

## What Was Implemented

### ✅ Completed Changes

1. **Real Conditioning System** (nodes.py:169-201)
   - Added `prepare_conditioning()` method
   - Prepares RGB latents in ComfyUI format for Flux
   - Includes task metadata for debugging

2. **Real Denoising Loop** (nodes.py:254-288)
   - Replaced alpha-blend placeholder with `comfy.sample.sample()`
   - Uses euler sampler with normal scheduler
   - Integrates with ComfyUI's sampling infrastructure
   - Active LoRA adapters influence denoising process
   - Full error handling with fallback

3. **Tunable Parameters** (nodes.py:420-433)
   - `noise_strength`: Control initial noise (default: 0.1)
   - `cfg_scale`: Classifier-free guidance (default: 1.0)
   - Both exposed as node inputs with tooltips

4. **Enhanced Logging**
   - Latent range before/after noise
   - Denoising progress
   - Output statistics
   - Error details

---

## Quick Test (5 minutes)

### Prerequisites

1. **Flux Model**: `flux1-dev.safetensors` in `ComfyUI/models/checkpoints/`
2. **OmniX Adapters**: Downloaded to `ComfyUI/models/loras/omnix/`
3. **Test Image**: Any panorama (2:1 aspect ratio) or regular image

### Steps

1. **Restart ComfyUI** (critical - reload Python code)
   ```bash
   # Kill ComfyUI completely, then restart
   python main.py
   ```

2. **Load Test Workflow**
   - Load `workflows/test_perception_simple.json`
   - Or manually create:
     - CheckpointLoaderSimple → load Flux
     - LoadImage → load test panorama
     - OmniXPanoramaValidator → fix aspect ratio
     - OmniXAdapterLoader → load adapters (bf16)
     - OmniXPanoramaPerception → connect all inputs
     - SaveImage → save outputs

3. **Configure for Fast Testing**
   - `num_steps`: **5** (for quick iteration)
   - `noise_strength`: 0.1
   - `cfg_scale`: 1.0
   - Enable only `extract_distance` (faster)

4. **Run Workflow**
   - Click "Queue Prompt"
   - Watch console for debug output

5. **Check Results**
   - ✅ **Success**: No errors, output image different from input
   - ⚠️ **Warning**: Errors in console → see Troubleshooting below
   - ❌ **Failure**: Crash or output identical to input

---

## Expected Behavior

### Console Output (Success)

```
[Perception] Input image shape: torch.Size([1, 1024, 2048, 3]), dtype: torch.float32
[Perception] Encoding pixels shape: torch.Size([1, 1024, 2048, 3])
[Perception] Encoded to latents shape: torch.Size([1, 16, 128, 256])
[Perception] Injecting LoRA adapters into Flux model...
  ✓ Loaded distance adapter
[Perception] ✓ Adapters injected successfully
[Perception] Activated adapter: distance
[Perception] Condition latents shape: torch.Size([1, 16, 128, 256])
[Perception] Added noise with strength 0.1
  Latent range before noise: [-3.245, 4.123]
  Latent range after noise: [-3.891, 4.567]
[Perception] Prepared conditioning for task 'distance'
  Condition latents shape: torch.Size([1, 16, 128, 256])
[Perception] Running Flux denoising with 5 steps, CFG=1.0
[Perception] ✓ Denoising completed successfully
  Output latent shape: torch.Size([1, 16, 128, 256])
  Output latent range: [-2.345, 3.890]
[Perception] Decoding latents shape: torch.Size([1, 16, 128, 256])
[Perception] Decoded to images shape: torch.Size([1, 1024, 2048, 3])
```

### Visual Output (What to Look For)

#### Distance Map
- Should show **depth variation** (not uniform gray)
- Bright/dark patterns corresponding to near/far regions
- May look noisy with 5 steps (expected)
- Should be **different** from input panorama

#### Normal Map
- Should show **colored surfaces** (RGB = XYZ normals)
- Not just a tinted version of input
- May show surface orientation patterns

#### Albedo Map
- Should look like **base colors** without shading
- Flatter lighting than input image
- May be subtle difference with 5 steps

---

## Progressive Testing

### Phase A: Verify Denoising Works (5 steps)

**Goal**: Confirm no crashes, output differs from input

1. Set `num_steps = 5`
2. Enable only `extract_distance`
3. Run and check for errors
4. Verify output is not identical to input

**Success Criteria**:
- ✅ No crashes or Python errors
- ✅ Output image has different pixel values than input
- ✅ Execution time < 10 seconds

---

### Phase B: Test Different Tasks (5 steps each)

**Goal**: Verify all adapters work

1. Test `distance` only
2. Test `normal` only
3. Test `albedo` only
4. Test all three together

**Success Criteria**:
- ✅ Each task completes without errors
- ✅ Outputs look different from each other
- ✅ Memory doesn't continuously increase

---

### Phase C: Increase Quality (10, 20, 28 steps)

**Goal**: Find optimal step count

1. Test with 10 steps
2. Test with 20 steps
3. Test with 28 steps (OmniX default)

**What to Observe**:
- Output quality should improve with more steps
- Depth maps should show clearer depth separation
- Normal maps should show more detail
- Execution time increases linearly

**Typical Times** (RTX 4090, 1024×2048):
- 5 steps: ~3-5 seconds
- 10 steps: ~6-10 seconds
- 20 steps: ~12-20 seconds
- 28 steps: ~17-28 seconds

---

### Phase D: Parameter Tuning

**Goal**: Find optimal noise_strength and cfg_scale

#### Test Noise Strength
Try: `0.05, 0.1, 0.15, 0.2` (keep num_steps=28, cfg_scale=1.0)

**Expected**:
- Too low (0.05): Output very similar to input, minimal perception
- Good (0.1-0.15): Clear perception outputs
- Too high (0.2+): Noisy/garbled outputs

#### Test CFG Scale
Try: `0.5, 1.0, 1.5, 2.0` (keep num_steps=28, noise_strength=0.1)

**Expected**:
- Low (0.5): Subtle perception, softer outputs
- Good (1.0-1.5): Clear perception
- High (2.0+): May over-emphasize features

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Cause**: Flux model + adapters + denoising exceeds VRAM

**Solutions**:
1. Use smaller resolution (512×1024 instead of 2048×1024)
2. Use fp8 Flux model instead of fp16/bf16
3. Enable model offloading in ComfyUI settings
4. Close other GPU applications
5. Test one task at a time

---

### Issue: "Conditioning format error"

**Error**: `Expected list, got torch.Tensor` or similar

**Cause**: Conditioning format doesn't match Flux expectations

**Debug Steps**:
1. Check console for full error traceback
2. Look for error in nodes.py around line 260
3. May need to adjust conditioning structure in `prepare_conditioning()`

**Temporary Fix**: The fallback will activate, returning input as output

---

### Issue: Output identical to input

**Symptoms**: All outputs look exactly like the input panorama

**Possible Causes**:
1. **LoRA not injecting**: Check for "Injecting LoRA adapters" message
2. **LoRA not active**: Check for "Activated adapter: X" message
3. **Noise too low**: Try increasing noise_strength to 0.2
4. **Steps too few**: Try 20-28 steps instead of 5
5. **Adapter weights missing**: Check adapters loaded successfully

**Debug**:
```python
# Check if adapters actually loaded
print(f"Adapters injected: {pipeline.adapters_injected}")
```

---

### Issue: Output is random noise

**Symptoms**: Garbage/static instead of perception maps

**Possible Causes**:
1. **Too much noise**: Reduce noise_strength to 0.05
2. **Too few steps**: Increase to 50+ steps
3. **Wrong conditioning**: May need to adjust conditioning format
4. **Adapter mismatch**: Wrong adapter for task

**Try**:
- noise_strength = 0.05
- num_steps = 50
- cfg_scale = 2.0

---

### Issue: "Cannot find adapter file"

**Error**: `FileNotFoundError: rgb_to_depth_depth.safetensors not found`

**Solution**:
1. Verify adapters downloaded:
   ```bash
   ls ComfyUI/models/loras/omnix/
   # Should show: rgb_to_depth_depth.safetensors, etc.
   ```

2. If missing, download:
   ```bash
   cd ComfyUI/custom_nodes/ComfyUI-OmniX
   python download_models.py
   ```

3. Check file permissions (Windows: not read-only)

---

### Issue: Very slow execution (>60s per task)

**Causes**:
1. Model on CPU instead of GPU
2. Heavy CPU offloading
3. Very high resolution

**Solutions**:
1. Check GPU usage during execution
2. Reduce resolution to 512×1024
3. Disable offloading if you have enough VRAM
4. Use bf16 precision instead of fp32

---

## Success Metrics

### Minimum Viable (Phase 1 Complete)

- ✅ Pipeline runs without crashing
- ✅ Outputs differ from input
- ✅ Distance maps show some depth variation
- ✅ Execution time reasonable (<30s at 1024×2048, 28 steps)
- ✅ All 5 tasks can run

### Good Quality (Phase 2 Target)

- ✅ Distance maps show clear near/far separation
- ✅ Normal maps show surface structure details
- ✅ Albedo maps show clean base colors
- ✅ PBR maps show material properties
- ✅ Outputs visually match OmniX paper examples

---

## Next Steps After Testing

1. **If Tests Pass**:
   - Document optimal parameters found
   - Test on multiple panoramas
   - Compare with official OmniX (if available)
   - Move to Phase 2: Quality validation

2. **If Tests Fail**:
   - Share error logs and console output
   - Provide test images and settings used
   - We'll debug and iterate

---

## Test Checklist

Use this to track testing progress:

### Basic Functionality
- [ ] ComfyUI loads nodes without errors
- [ ] Workflow connects all inputs
- [ ] Model and VAE load successfully
- [ ] Adapters load successfully
- [ ] Pipeline runs without crashes

### Denoising
- [ ] Denoising loop executes (no fallback)
- [ ] 5 steps complete in <10s
- [ ] 28 steps complete in <30s
- [ ] No memory leaks (can run multiple times)

### Output Quality
- [ ] Distance output differs from input
- [ ] Normal output differs from input
- [ ] Albedo output differs from input
- [ ] Outputs show perception data (not just noise)

### Parameters
- [ ] Tested noise_strength: 0.05, 0.1, 0.2
- [ ] Tested cfg_scale: 0.5, 1.0, 2.0
- [ ] Tested steps: 5, 10, 20, 28
- [ ] Found optimal parameters

### Multiple Scenarios
- [ ] Indoor panorama
- [ ] Outdoor panorama
- [ ] Different resolutions (512, 1024, 2048)
- [ ] Single task vs all tasks

---

## Contact

If you encounter issues not covered here:
1. Check console output for full error trace
2. Note exact parameters used
3. Share test image if possible
4. Report via GitHub issues or discussion

---

**Remember**: This is Phase 1 testing. The goal is to verify the denoising loop works, not perfect quality. Outputs may be rough with low step counts - that's expected and normal!
