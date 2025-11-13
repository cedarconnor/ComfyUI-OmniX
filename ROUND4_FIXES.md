# Round 4 Fixes - CLIP Conditioning

**Date**: January 12, 2025
**Issue**: Flux requires 3D text embeddings, not 4D image latents

---

## Problem Identified

**Error**: `ValueError: Input img and txt tensors must have 3 dimensions.`

**Root Cause**: Flux's forward pass expects text embeddings (3D: `(B, seq_len, dim)`) but we were passing image latents (4D: `(B, C, H, W)`).

Flux is a text-to-image model that requires proper text conditioning.

---

## Solution: Use CLIP for Text Conditioning

Instead of using image latents as conditioning, we now:
1. Use CLIP to encode a text prompt
2. The prompt describes the perception task
3. LoRA adapters should override the text guidance

---

## Changes Applied

### 1. Added CLIP to Pipeline ✅
**File**: `nodes.py` lines 38-64

```python
def __init__(
    self,
    vae: VAE,
    model: Any,
    clip: Any,  # <-- NEW
    adapter_manager: AdapterManager,
    ...
):
    self.clip = clip
```

### 2. CLIP Text Encoding ✅
**File**: `nodes.py` lines 253-271

```python
# Encode perception prompt with CLIP
tokens = self.clip.tokenize(f"perception task: {task}")
cond, pooled = self.clip.encode_from_tokens(tokens, return_pooled=True)

positive_cond = [[cond, {"pooled_output": pooled}]]

# Empty negative prompt
empty_tokens = self.clip.tokenize("")
neg_cond, neg_pooled = self.clip.encode_from_tokens(empty_tokens, return_pooled=True)
negative_cond = [[neg_cond, {"pooled_output": neg_pooled}]]
```

### 3. Updated Node Inputs ✅
**File**: `nodes.py` lines 394-413

Added CLIP as required input:
```python
"required": {
    "model": ("MODEL", ...),
    "clip": ("CLIP", ...),  # <-- NEW
    "vae": ("VAE", ...),
    ...
}
```

### 4. Updated Workflow ✅
**File**: `workflows/test_perception_simple.json`

- Connected CLIP from CheckpointLoader to OmniXPanoramaPerception
- Link 9: CLIP connection

---

## How It Works Now

```
CheckpointLoaderSimple
├─ MODEL → OmniXPanoramaPerception
├─ CLIP → OmniXPanoramaPerception  (NEW!)
└─ VAE → OmniXPanoramaPerception

OmniXPanoramaPerception:
1. Encode RGB panorama → latents (for LoRA conditioning)
2. Encode text with CLIP → text embeddings (for Flux)
3. Run Flux denoising with:
   - Text conditioning from CLIP
   - LoRA adapter active (should override text)
   - Output: perception latents
4. Decode → perception images
```

---

## Expected Behavior

### Text Conditioning Role

The text prompt is: `"perception task: {task}"` where `{task}` is distance/normal/albedo/etc.

**Important**: The text conditioning is mainly to satisfy Flux's requirements. The LoRA adapters should be the primary guidance mechanism. The text just provides a semantic hint.

### LoRA Should Dominate

With the LoRA adapter active and properly injected, it should override most of the text conditioning influence and drive the model toward perception outputs.

---

## Testing

### Expected Console Output

```
[Perception] Running distance perception with 5 steps
[Perception] Encoding text conditioning with CLIP...
  Text cond shape: torch.Size([1, 77, 4096]), pooled shape: torch.Size([1, 768])
[Perception] Running Flux denoising with 5 steps, CFG=1.0
[Perception] ✓ Denoising completed successfully  # <-- Should see this!
```

### Success Criteria

- ✅ No "Input img and txt tensors must have 3 dimensions" error
- ✅ Denoising completes successfully
- ✅ Output images produced
- ⚠️ Output quality depends on LoRA injection (still at 0 layers)

---

## Next Steps

### If Denoising Works ✅

**Great!** The pipeline is functional. Next priority:

**Fix LoRA Injection** - We know the Flux structure now:
```
double_blocks[i].img_attn  <- inject LoRA here
single_blocks[i].linear1   <- inject LoRA here
single_blocks[i].linear2   <- inject LoRA here
```

Update `omnix/cross_lora.py` to target these specific modules.

### If Still Fails ⚠️

Check error message:
- Different text shape issue → adjust tokenization
- Memory error → reduce resolution
- Other Flux error → may need different approach

---

## Why This Approach

### Why Not Use Image Latents as Conditioning?

Flux's architecture requires text embeddings in specific format. Image latents are 4D, text is 3D - they're incompatible.

### Why Will This Work?

1. **Flux gets proper text conditioning** (satisfies its requirements)
2. **LoRA adapters are active** (modifies attention behavior)
3. **RGB latents in the denoising start point** (provides scene information)

The combination of all three should produce perception outputs.

---

## Potential Issues

### Issue: LoRA Still Not Working (0 Patches)

**Status**: Known issue, will fix after denoising works

**Impact**: Without LoRA active, output will be text-guided image, not perception

**Next**: Update cross_lora.py patching logic

### Issue: Output Doesn't Look Like Perception

**Possible Causes**:
1. LoRA not injected (0 patches) - most likely
2. LoRA not strong enough
3. Text conditioning too strong
4. Need different sampling parameters

**Debug**:
- First check if LoRA patching works
- Then tune adapter strength / CFG scale

---

## Files Modified

1. **nodes.py**
   - Added CLIP parameter throughout
   - Changed conditioning to use CLIP encoding
   - Updated node INPUT_TYPES

2. **workflows/test_perception_simple.json**
   - Added CLIP connection from checkpoint to perception node

---

**Status**: Ready for Round 4 test!

**Action**: Restart ComfyUI, load updated workflow, run test

**Watch For**:
- ✅ Text conditioning shapes in console
- ✅ Denoising completion message
- 📊 Output images (even if not perfect perception yet)
