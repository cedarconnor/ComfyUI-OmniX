# Round 3 Fixes - Conditioning & LoRA Debug

**Date**: January 12, 2025
**Issues**: Missing pooled_output, LoRA not finding layers

---

## Fix 1: Added pooled_output to Conditioning ✅

**Error from Round 2**:
```
KeyError: 'pooled_output'
at comfy\model_base.py:882: return kwargs["pooled_output"]
```

**Root Cause**: Flux requires `pooled_output` in conditioning dict, we were passing empty `{}`

**Fix Applied** (nodes.py:250-259):
```python
# Before
positive_cond = [[condition_latents, {}]]
negative_cond = [[torch.zeros_like(condition_latents), {}]]

# After
pooled = condition_latents.mean(dim=[2, 3])  # Average over spatial dimensions
positive_cond = [[condition_latents, {"pooled_output": pooled}]]
negative_cond = [[torch.zeros_like(condition_latents), {"pooled_output": torch.zeros_like(pooled)}]]
```

**Why This Works**:
- Flux expects a pooled representation of the condition
- We create it by averaging the latents over spatial dimensions
- Shape: (B, 16, H, W) → (B, 16) via mean over H and W
- This satisfies Flux's requirement

---

## Fix 2: Enhanced LoRA Debug Logging ✅

**Issue**: Still `Patched 0 layers` but we need to see WHY

**Debug Info from Round 2**:
```
[Cross-LoRA Debug] Model type: <class 'comfy.model_patcher.ModelPatcher'>
[Cross-LoRA Debug] Transformer type: <class 'comfy.ldm.flux.model.Flux'>
[Cross-LoRA Debug] Transformer has 9 children
```

**Fix Applied** (cross_lora.py:133-142):
```python
# List all children to see Flux structure
children = list(transformer.named_children())
print(f"[Cross-LoRA Debug] Transformer has {len(children)} children:")
for name, child in children:
    print(f"  - {name}: {type(child).__name__}")
    # If it's a ModuleList (like double_blocks or single_blocks), show first element
    if type(child).__name__ == 'ModuleList' and len(child) > 0:
        first_block = child[0]
        block_children = list(first_block.named_children())
        print(f"    Block[0] has {len(block_children)} children: {[n for n, _ in block_children[:5]]}")
```

**What This Will Show**:
- Names of the 9 children (likely: `img_in`, `double_blocks`, `single_blocks`, etc.)
- Structure of blocks (where the attention layers actually are)
- Exact path to target for LoRA injection

---

## Expected Results from Round 3

### Denoising Should Work Now ✅

With pooled_output added, the conditioning should be accepted by Flux:

```
[Perception] Running Flux denoising with 5 steps, CFG=1.0
[Perception] ✓ Denoising completed successfully  # <-- Should see this!
  Output latent shape: torch.Size([1, 16, 128, 256])
  Output latent range: [...]
```

### LoRA Debug Should Show Structure 🔍

```
[Cross-LoRA Debug] Transformer has 9 children:
  - img_in: Linear
  - time_in: MLPEmbedder
  - double_blocks: ModuleList  # <-- This is where attn layers are!
    Block[0] has 5 children: ['img_attn', 'txt_attn', 'img_mlp', ...]
  - single_blocks: ModuleList  # <-- And here!
    Block[0] has 3 children: [...]
  ...
```

Then we can update the patching logic to target the right modules.

---

## What Should Happen

### Best Case ✅
1. Denoising works! (no KeyError)
2. Debug shows Flux structure clearly
3. Output is different from input (even without LoRA active yet)
4. We can then fix LoRA injection based on debug output

### Likely Case ⚠️
1. Denoising works! ✅
2. Debug shows structure ✅
3. LoRA still 0 patches (expected, need to update patching logic) ⚠️
4. Output different from input but not full perception yet
5. Next round: Fix LoRA injection to target correct modules

---

## Next Steps After Round 3

### If Denoising Works ✅

Great! The core pipeline is functional. Next:

1. **Analyze debug output** to find where attention layers are
2. **Update `patch_module()` function** to target correct Flux structure
3. **Target likely modules**:
   - `double_blocks[i].img_attn.qkv` or similar
   - `single_blocks[i].linear1` or similar
   - Need to see actual structure first

### If Denoising Still Fails ⚠️

Try alternative approaches:
1. Use only latent image without conditioning
2. Set CFG to 0.0 (no conditioning guidance)
3. Use img2img mode with high denoise
4. Create minimal test case

---

## Technical Details

### Pooled Output Shape

```python
condition_latents: (B, 16, 128, 256)  # Latent image
pooled: (B, 16)                        # Spatial mean
```

This matches what Flux expects from CLIP text encoder (though we're using image latents instead of text).

### Flux Model Structure (Expected)

Based on official Flux implementation:
```
Flux
├── img_in: Linear (projects input)
├── time_in: MLPEmbedder (timestep embedding)
├── txt_in: Linear (text embedding projection)
├── vector_in: Linear (vector embedding)
├── guidance_in: Linear (guidance embedding)
├── double_blocks: ModuleList[N]
│   └── [0..N-1]: DoubleStreamBlock
│       ├── img_attn: Attention (image self-attention)
│       ├── txt_attn: Attention (text cross-attention)
│       ├── img_mlp: MLP
│       └── txt_mlp: MLP
├── single_blocks: ModuleList[M]
│   └── [0..M-1]: SingleStreamBlock
│       ├── linear1: Linear (combined qkv)
│       └── linear2: Linear (output projection)
├── final_layer: Linear
└── ...
```

We need to inject LoRA into:
- `double_blocks[i].img_attn.to_q/k/v`
- `single_blocks[i].linear1`

---

## Files Changed

1. **nodes.py**
   - Lines 250-259: Added pooled_output to conditioning

2. **omnix/cross_lora.py**
   - Lines 133-142: Enhanced debug logging for model structure

---

## Testing Instructions

**Restart ComfyUI and run same workflow**

Watch for:
1. ✅ No KeyError on pooled_output
2. ✅ Denoising completes successfully
3. 🔍 Debug output showing Flux children structure
4. 📊 Output image different from input (even without LoRA)

---

**Status**: Ready for Round 3 test!
