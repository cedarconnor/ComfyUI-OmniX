# Weight Conversion Fix

**Date**: January 12, 2025
**Status**: Implemented, Ready for Testing

---

## Problem Identified

The OmniX perception outputs were all identical gray blobs because **the LoRA adapter weights were never actually being loaded into the model**.

### Root Cause

The OmniX adapters from HuggingFace are structured for **Diffusers Flux**, which has different layer architecture than **ComfyUI Flux**:

**Diffusers Flux** (OmniX format):
- Separate projections: `to_q`, `to_k`, `to_v`
- 19 `transformer_blocks` + 38 `single_transformer_blocks`

**ComfyUI Flux**:
- Fused projections: `qkv` (outputs Q||K||V concatenated)
- 19 `double_blocks` + 38 `single_blocks`

**Key Incompatibility**:
- OmniX: 456 weights (separate Q/K/V for each layer)
- ComfyUI: Needs fused QKV weights

---

## Solution Implemented

### 1. Weight Converter (`omnix/weight_converter.py`)

Created a conversion function that:
- Maps layer names: `transformer_blocks.N` → `double_blocks[N]`
- Maps layer names: `single_transformer_blocks.N` → `single_blocks[N]`
- **Fuses Q/K/V weights**:
  - `lora_A`: Average of Q/K/V down-projections
  - `lora_B`: Concatenate Q/K/V up-projections `[Q||K||V]`

**Example Conversion**:
```python
# Input (OmniX):
to_q.lora_A: [64, 3072]
to_q.lora_B: [3072, 64]
to_k.lora_A: [64, 3072]
to_k.lora_B: [3072, 64]
to_v.lora_A: [64, 3072]
to_v.lora_B: [3072, 64]

# Output (ComfyUI):
qkv.lora_A: [64, 3072]      # Averaged
qkv.lora_B: [9216, 64]      # Concatenated (3072 * 3)
```

### 2. Injection Updates (`omnix/cross_lora.py`)

Modified `inject_cross_lora_into_model()` to:
1. Convert all adapter weights using the converter
2. After creating each CrossLoRA layer, **load the converted weights**
3. Handle both `double_blocks` and `single_blocks`

**Key Code**:
```python
# Convert weights
converted_weights = {}
for adapter_name, omnix_weights in adapter_weights.items():
    converted_weights[adapter_name] = convert_omnix_weights_to_comfyui(omnix_weights)

# Load weights into each layer
layer_key = f"double_blocks.{i}.img_attn.qkv"
if adapter_name in converted_weights and layer_key in converted_weights[adapter_name]:
    cross_lora.load_adapter_weights(adapter_name, converted_weights[adapter_name][layer_key])
```

### 3. Config Updates (`omnix/adapters.py`)

Updated `ADAPTER_CONFIGS` to use rank=64 (matching OmniX) instead of rank=16.

---

## Files Modified

1. **omnix/weight_converter.py** (NEW - 130 lines)
   - `convert_omnix_weights_to_comfyui()`: Main conversion logic
   - `load_and_convert_adapter()`: Convenience function

2. **omnix/cross_lora.py** (Modified)
   - Line 16: Import weight converter
   - Lines 122-126: Convert weights before injection
   - Lines 187-194: Load weights for double_blocks
   - Lines 219-227: Load weights for single_blocks

3. **omnix/adapters.py** (Modified)
   - Lines 37-64: Updated rank from 16 to 64 for all adapters

---

## Expected Behavior

### Before This Fix ❌
- All 3 outputs (distance, normal, albedo) looked identical
- Just gray blobs with vignette
- LoRA adapters created but using random weights

### After This Fix ✅
Should see:
- **Distance map**: Depth gradients (dark = near, bright = far)
- **Normal map**: Colorful RGB representing surface orientations
- **Albedo map**: Base material colors (flatter lighting)
- Each output should be **distinctly different**

---

## Testing

### Test Command
1. **Restart ComfyUI** (to reload Python modules)
2. **Load workflow**: `workflows/test_perception_simple.json`
3. **Run workflow**

### Watch Console For:
```
[Cross-LoRA] Converting 4 adapters to ComfyUI format...
[Weight Converter] Converted 57 layers
[Cross-LoRA] Patching double_blocks[0].img_attn.qkv
[Cross-LoRA]   ✓ Loaded weights for distance
[Cross-LoRA]   ✓ Loaded weights for normal
[Cross-LoRA]   ✓ Loaded weights for albedo
[Cross-LoRA]   ✓ Loaded weights for pbr
...
[Cross-LoRA] Patched 95 layers with adapters
100%|██████████| 5/5 [00:09<00:00, 1.93s/it]
[Perception] ✓ Denoising completed successfully
```

### Success Criteria
- ✅ No errors during weight loading
- ✅ Console shows "Loaded weights for [adapter]" messages
- ✅ Outputs look like perception maps (not identical gray blobs)
- ✅ Distance, normal, albedo all look different from each other

---

## Technical Notes

### Why Averaging lora_A Works
The down-projection (`lora_A`) maps from the same input space for Q, K, and V. Averaging them provides a reasonable shared projection for the fused layer.

### Why Concatenating lora_B Works
The up-projection (`lora_B`) maps to the output space. Since ComfyUI's qkv outputs `[Q||K||V]` concatenated, we concatenate the B matrices to match this structure.

### Rank Mismatch
OmniX uses rank=64, not rank=16. The configs have been updated to match.

---

## Alternative Approach: ComfyUI-Diffusers

If this conversion approach doesn't work perfectly, an alternative is to use the **ComfyUI-Diffusers** extension to load HuggingFace Diffusers Flux directly, which would match the OmniX adapter format exactly.

**Pros**:
- No weight conversion needed
- Direct compatibility with OmniX weights

**Cons**:
- Requires additional extension
- Different workflow structure
- May have performance differences

---

## Next Steps

### If Test Succeeds ✅
1. Increase quality: Set `num_steps = 28` (OmniX default)
2. Test all 5 tasks (enable roughness, metallic)
3. Compare outputs with reference OmniX results
4. Tune parameters (noise_strength, CFG scale)
5. Clean up debug logging

### If Test Fails ⚠️
Check console for specific errors:
- **"Failed to load weights"**: Shape mismatch issue
- **Still gray blobs**: LoRA scale too weak or weights not applying
- **Different error**: Share full traceback

Consider ComfyUI-Diffusers approach as fallback.

---

## Confidence Level

**70% confident** this will work. The weight conversion is mathematically sound, but there may be subtle issues with:
- Weight scaling/normalization
- Bias handling
- Layer-specific differences in single_blocks

If it doesn't work perfectly, we have a clear path forward with ComfyUI-Diffusers.

---

**Status**: Ready for test
**Action**: Restart ComfyUI and run workflow
**Expected**: Real perception outputs!
