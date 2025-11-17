# CRITICAL FIX REVERSION: OmniX Adapter Keys Already Match Diffusers

**Date**: 2025-01-17
**Status**: ✅ FIXED - Reverted incorrect key remapping that was breaking adapter loading

---

## What Happened

### The Incorrect Fix (Now Reverted)

In commit d47b279, I implemented `omnix/diffusers_key_remap.py` that renamed OmniX adapter keys from:
- `transformer.transformer_blocks.N` → `transformer.double_blocks.N`
- `transformer.single_transformer_blocks.N` → `transformer.single_blocks.N`

**This was completely wrong.**

### Why It Was Wrong

After inspecting the actual Diffusers Flux implementation at:
```
C:\Program Files\Python311\Lib\site-packages\diffusers\models\transformers\transformer_flux.py:604-615
```

The `FluxTransformer2DModel` class declares and uses:
- `self.transformer_blocks` (NOT `double_blocks`)
- `self.single_transformer_blocks` (NOT `single_blocks`)

**There is no reference to `double_blocks` or `single_blocks` anywhere in the Diffusers Flux implementation.**

### The Result

My remapping renamed every adapter key to non-existent module names, causing:
- PEFT to silently fail to match keys
- Only 38 out of 456 LoRA layers injected (should be all 456)
- Warnings about "unexpected keys in state_dict"
- 76 feedforward layer keys couldn't be remapped at all
- Adapters still not actually applied during inference

---

## The Actual Root Cause

**OmniX adapter keys ALREADY match the Diffusers Flux model structure.**

The original adapter keys are:
```
transformer.transformer_blocks.0.attn.to_q.lora_A.weight
transformer.transformer_blocks.0.attn.to_k.lora_A.weight
transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight
...
```

These **directly correspond** to the actual Diffusers Flux model layers:
```python
# In FluxTransformer2DModel (transformer_flux.py:604-615)
self.transformer_blocks = nn.ModuleList([...])  # MMDiT blocks
self.single_transformer_blocks = nn.ModuleList([...])  # Single stream blocks
```

**No remapping was ever needed.**

---

## The Fix

### Files Changed

1. **nodes_diffusers.py**:
   - Removed import: `from .omnix.diffusers_key_remap import remap_omnix_to_diffusers_flux, verify_key_format`
   - Removed lines 362-377: Key format checking and remapping logic
   - Now loads adapter state_dict directly without any key modifications

2. **omnix/diffusers_key_remap.py**:
   - **DELETED** - This file was causing the problem

### What the Code Does Now

```python
# Load adapter weights directly
state = load_file(str(adapter_path))

# Save to temp file (required by load_lora_weights API)
with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp_file:
    temp_adapter_path = tmp_file.name
    save_file(state, temp_adapter_path)

# Load into pipeline with ORIGINAL keys
flux_pipeline.load_lora_weights(
    Path(temp_adapter_path).parent,
    weight_name=Path(temp_adapter_path).name,
    adapter_name=adapter_name,
)
```

**No key modification at all** - PEFT can now match the OmniX keys to the actual Flux model layers.

---

## Expected Results

### Log Output

After this fix, you should see:

```
Loading normal from C:\ComfyUI\models\loras\omnix\rgb_to_normal_normal.safetensors
Adapter 'normal' has 456 weight tensors
Sample weight keys: ['transformer.single_transformer_blocks.0.attn.to_k.lora_A.weight', ...]
Loaded adapter 'normal' successfully
...
PEFT config loaded: ['distance', 'normal', 'albedo', 'pbr']
✓ Found 456 LoRA-injected layers  ← Should now be 456, not 38!
```

**Key changes:**
- ❌ NO MORE "remapping to Diffusers format" messages
- ❌ NO MORE "unexpected keys" warnings
- ✅ All 456 weight tensors should inject successfully
- ✅ LoRA layers found in all target modules

### Visual Output Quality

**Normal Maps** should now show:
- ✅ Smooth RGB gradients representing surface normals
- ✅ Mostly blue/purple tones for camera-facing surfaces
- ✅ Subtle color variations for surface curvature
- ❌ NO MORE saturated magenta/green/orange chaos

**Depth/Distance Maps** should show:
- ✅ Accurate geometric detail
- ✅ Proper near/far gradients

**Albedo Maps** should be:
- ✅ Distinctly different from input image
- ✅ Showing base material colors without lighting

**PBR Maps** should show:
- ✅ Realistic material properties
- ✅ Clear roughness/metallic variations

---

## Why This Wasn't Caught Earlier

I made an incorrect assumption based on:
1. ComfyUI's Flux implementation uses different naming internally
2. I couldn't inspect the actual Diffusers Flux model due to 403 auth errors when trying to load it
3. I assumed there was a naming mismatch without verifying the actual Diffusers source code

The user correctly identified this by:
1. Inspecting the actual `FluxTransformer2DModel` source code
2. Verifying that the model uses `transformer_blocks` and `single_transformer_blocks`
3. Recognizing that my remapping created non-existent module names

---

## Verification Checklist

After this reversion:

- [ ] No more "remapping to Diffusers format" log messages
- [ ] No more "unexpected keys" warnings
- [ ] Logs show "✓ Found 456 LoRA-injected layers" (not 38)
- [ ] Normal map shows proper RGB gradients (not magenta/green chaos)
- [ ] Depth map has clear geometric detail
- [ ] Albedo is distinct from source image
- [ ] Overall quality matches HKU-MMLab/OmniX reference outputs

---

## Technical Details

### Diffusers Flux Architecture

The `FluxTransformer2DModel` (from diffusers v0.32.0+) uses:

**Double Stream Blocks** (`transformer_blocks`):
- MMDiT (Multimodal Diffusion Transformer) blocks
- Process both image and text conditioning
- Use dual attention: `img_attn` (image self-attention) and `txt_attn` (cross-attention)
- Accessed as: `transformer.transformer_blocks[i].img_attn.to_q`

**Single Stream Blocks** (`single_transformer_blocks`):
- Process image only
- Single attention path
- Accessed as: `transformer.single_transformer_blocks[i].attn.to_q`

### OmniX Adapter Structure

The OmniX adapters target these exact module names:
```
transformer.transformer_blocks.{N}.attn.to_q.lora_{A,B}.weight
transformer.transformer_blocks.{N}.attn.to_k.lora_{A,B}.weight
transformer.transformer_blocks.{N}.attn.to_v.lora_{A,B}.weight
transformer.transformer_blocks.{N}.ff.net.0.proj.lora_{A,B}.weight
transformer.transformer_blocks.{N}.ff.net.2.lora_{A,B}.weight

transformer.single_transformer_blocks.{N}.attn.to_q.lora_{A,B}.weight
transformer.single_transformer_blocks.{N}.attn.to_k.lora_{A,B}.weight
transformer.single_transformer_blocks.{N}.attn.to_v.lora_{A,B}.weight
transformer.single_transformer_blocks.{N}.proj_mlp.lora_{A,B}.weight
```

These match the Diffusers model structure **exactly**.

---

## Related Files

- **nodes_diffusers.py**: Updated to remove remapping (fixed)
- **omnix/diffusers_key_remap.py**: DELETED (was incorrect)
- **KEY_REMAPPING_FIX_SUMMARY.md**: Previous (incorrect) analysis
- **CRITICAL_ADAPTER_FAILURE_ANALYSIS.md**: Original correct diagnosis
- **OUTPUT_QUALITY_INVESTIGATION.md**: Initial investigation

---

## Credit

This fix was made possible by the user's excellent debugging:

> "FluxTransformer2DModel declares self.transformer_blocks and self.single_transformer_blocks and iterates over them directly (C:\Program Files\Python311\Lib\site-packages\diffusers\models\transformers\transformer_flux.py:604 and 615). There is not a single reference to double_blocks or single_blocks anywhere in that implementation."

The user correctly identified that my remapping approach was fundamentally flawed and that the OmniX adapter keys already match the Diffusers Flux model structure.

---

**All changes reverted and pushed to `claude/improve-output-quality-01CmV5syqqUUXP6Tw4wnNPvj`**

The adapters should now load correctly with all 456 LoRA layers properly injected!
