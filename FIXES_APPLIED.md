# Quick Fixes Applied - Round 2

**Date**: January 12, 2025
**Issues Found**: LoRA injection failing, conditioning format wrong

---

## Issues from First Test

### Issue 1: LoRA Not Injecting ❌
```
[Cross-LoRA] Patched 0 layers with adapters
```

**Root Cause**: The injection code couldn't find the right model structure to patch

**Fix Applied**:
- Added extensive debug logging to `cross_lora.py` lines 112-132
- Will show model structure in next test
- Can then update patch logic to match actual Flux structure

---

### Issue 2: Denoising Call Failed ❌
```
[Perception] ✗ Denoising failed: 'NoneType' object is not iterable
```

**Root Cause**: `comfy.sample.sample()` has different signature/requirements than expected

**Fix Applied** (nodes.py:250-289):
1. Simplified conditioning format:
   ```python
   # Old: Complex conditioning dict
   # New: Simple format
   positive_cond = [[condition_latents, {}]]
   negative_cond = [[torch.zeros_like(condition_latents), {}]]
   ```

2. Switched to `nodes.common_ksampler()`:
   ```python
   # Old: comfy.sample.sample(...)  # Wrong signature
   # New: nodes.common_ksampler(...) # Standard ComfyUI approach
   ```

3. Added better error traceback to see full exception details

**Why This Should Work**:
- `common_ksampler` is what KSampler node uses internally
- Simpler, more reliable API
- Standard format for conditioning

---

## Expected Results from Next Test

### Best Case ✅
```
[Cross-LoRA Debug] Model type: <class ...>
[Cross-LoRA Debug] Transformer has X children
[Cross-LoRA] Patched X layers with adapters  # X > 0!
[Perception] Running Flux denoising...
[Perception] ✓ Denoising completed successfully
```

### Worst Case ⚠️
```
[Cross-LoRA] Patched 0 layers  # Still 0
[Perception] ✗ Denoising failed: <new error>
```

If LoRA still not patching:
- Debug output will show us the actual Flux model structure
- We can then fix the patching logic to match
- May need to target different layer names

If denoising still fails:
- Traceback will show exact error
- Conditioning might need different format for Flux
- May need to use different sampling approach

---

## Changes Summary

### Files Modified

1. **nodes.py**
   - Line 20: Added `import nodes` for common_ksampler
   - Line 18: Added `import comfy.samplers`
   - Lines 250-289: Rewrote denoising call
   - Simplified conditioning format
   - Better error logging

2. **omnix/cross_lora.py**
   - Lines 112-132: Added debug logging
   - Shows model type, attributes, structure
   - Will help diagnose injection failure

---

## Testing Instructions

### Quick Re-test (2 minutes)

1. **Restart ComfyUI** (reload Python changes)

2. **Run same workflow** (test_perception_simple.json)

3. **Check console for**:
   - `[Cross-LoRA Debug]` messages showing model structure
   - LoRA patch count (hopefully > 0)
   - Denoising success or new error with traceback

---

## Next Steps

### If LoRA Patches (count > 0) and Denoising Works ✅
**SUCCESS!** Move to quality testing:
- Increase steps to 28
- Test all tasks
- Tune parameters
- Compare outputs

### If LoRA Still Doesn't Patch (count = 0) ⚠️
**Debug Required**:
1. Look at debug output showing model structure
2. Update `patch_module()` function to target correct layers
3. May need to search for different attribute names
4. Possible Flux model structure is very different

### If Denoising Still Fails ⚠️
**Alternative Approaches**:
1. Try text-based conditioning instead of latent conditioning
2. Use img2img approach with denoise parameter
3. Create minimal repro with just Flux + LoRA
4. May need to condition differently for Flux

---

## Code Changes Detail

### Conditioning Format Change

**Before**:
```python
conditioning = [[
    condition_latents,
    {
        "pooled_output": condition_latents,
        "task": task,
        "guidance": "perception"
    }
]]
```

**After**:
```python
positive_cond = [[condition_latents, {}]]
negative_cond = [[torch.zeros_like(condition_latents), {}]]
```

**Why**: Simpler format, matches what KSampler expects

---

### Sampling Function Change

**Before**:
```python
result = comfy.sample.sample(
    model=self.model,
    noise=noisy_latents,
    steps=num_steps,
    cfg=cfg_scale,
    sampler_name="euler",
    scheduler="normal",
    positive=conditioning,
    negative=None,
    latent_image=latent_dict,
    denoise=1.0,
    disable_noise=False,
    start_step=0,
    last_step=num_steps,
    force_full_denoise=True
)
latents = result[0]["samples"]
```

**After**:
```python
latents = nodes.common_ksampler(
    model=self.model,
    seed=42,
    steps=num_steps,
    cfg=cfg_scale,
    sampler_name="euler",
    scheduler="normal",
    positive=positive_cond,
    negative=negative_cond,
    latent=latent_image,
    denoise=1.0
)[0]["samples"]
```

**Why**: `common_ksampler` is the standard ComfyUI API

---

## Confidence Level

**Denoising Fix**: 90% confident this will work
- Using standard ComfyUI API now
- Proper conditioning format
- Matches KSampler implementation

**LoRA Injection**: 50% confident
- Need to see debug output first
- May require iteration to find right layers
- Flux model structure might be different than expected

---

## If Both Still Fail...

### Plan B: Simpler Approach
1. Don't use LoRA at all initially
2. Just test if denoising works without adapters
3. Add LoRA later once basic pipeline works
4. Focus on getting ANY different output first

### Plan C: Manual LoRA Application
1. Load adapter weights manually
2. Apply them directly to model parameters
3. Skip the injection framework
4. More brittle but might work

---

**Status**: Fixes applied, ready for round 2 testing

**Action**: Restart ComfyUI and run again

