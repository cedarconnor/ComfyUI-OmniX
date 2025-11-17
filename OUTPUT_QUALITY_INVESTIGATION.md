# Output Quality Investigation - ComfyUI-OmniX

**Date**: 2025-01-17
**Issue**: Generated perception outputs (depth, normal, albedo, PBR) do not match the quality of the original HKU-MMLab/OmniX repository

## Test Configuration (From User)

Based on `TestImages/test_perception_simple.json`:

```json
OmniXLoRALoader settings:
- adapter_dir: "C:/ComfyUI/models/loras/omnix"
- all adapters enabled: distance, normal, albedo, pbr
- lora_scale: 1.3  ⚠️

OmniXPerceptionDiffusers settings:
- num_steps: 28 ✓
- guidance_scale: 3.5 ✓
- noise_strength: 0.3 ✓
```

## Observations from Test Images

1. **Normal Map (`normal_00001_.png`)**:
   - Shows highly saturated magenta/green/blue colors
   - Does NOT resemble correct surface normals
   - Suggests either:
     - LoRA adapters not being applied correctly
     - Adapters being applied but with wrong parameters
     - Visualization issue (unlikely, given visualization code review)

2. **Depth Map (`depth_00001_.png`)**:
   - Appears grayscale (correct)
   - May lack geometric detail compared to reference

3. **Albedo Map (`albedo_00001_.png`)**:
   - Looks similar to source image
   - Suggests insufficient adapter influence

## Investigation Findings

### 1. LoRA Weight Loading Flow

**Code Analysis**: `nodes_diffusers.py` lines 362-402

```python
flux_pipeline.load_lora_weights(
    adapter_dir,
    weight_name=filename,
    adapter_name=adapter_name,
)
```

**Finding**: The code loads OmniX LoRA adapters **directly** via HuggingFace Diffusers' `load_lora_weights()` without any conversion.

**Implication**: This assumes that:
1. OmniX adapter weight keys match Diffusers' FluxPipeline layer names
2. PEFT integration will automatically hook the adapters into the transformer
3. No special weight manipulation is needed

### 2. Weight Converter Not Used for Diffusers Pipeline

**Code**: `omnix/weight_converter.py` contains block-diagonal LoRA fusion logic

**Code**: `omnix/cross_lora.py` line 16 imports and uses the converter

**Finding**: The weight converter is ONLY used by `cross_lora.py` (legacy ComfyUI implementation), NOT by `nodes_diffusers.py`.

**Implication**: The Diffusers pipeline code path bypasses the block-diagonal fusion fix that was implemented to preserve Q/K/V independence.

### 3. Adapter Activation During Inference

**Code**: `nodes_diffusers.py` line 607

```python
flux_pipeline.set_adapters([task], adapter_weights=[loaded_adapters[task]["scale"]])
```

**Finding**: Adapters are activated via `set_adapters()` before each perception run.

**Potential Issue**: If PEFT doesn't correctly map the OmniX adapter keys to Flux layer names, the adapters won't be applied during transformer forward passes.

### 4. Prompt Configuration

**Code**: `nodes_diffusers.py` line 638

```python
prompt = f"perception task: {task}, equirectangular projection, 360 degree panoramic view"
```

**Finding**: Using descriptive prompts for perception tasks.

**Potential Issue**: The original OmniX adapters may have been trained with:
- Empty prompts (`""`)
- Simple task tokens (e.g., `"depth"`, `"normal"`)
- No prompt at all (unconditional)

Using the wrong prompt can significantly degrade adapter effectiveness.

### 5. LoRA Scale Parameter

**User Setting**: `lora_scale=1.3`
**Typical Default**: `lora_scale=1.0`

**Finding**: User is applying 30% higher LoRA influence than default.

**Potential Issue**:
- High LoRA scale can cause instability and artifacts
- May amplify any errors in adapter application
- Original OmniX likely uses scale=1.0

## Root Cause Hypothesis

Based on the evidence, the most likely issues are (in order of probability):

### **Primary Suspect: Adapter Key Mismatch**

The OmniX adapters have weight keys following one naming convention (e.g., `transformer.transformer_blocks.N.attn.to_q.lora_A.weight`), but the actual Flux model in Diffusers might use different layer names.

**Evidence**:
- Diffusers PEFT requires exact key matches to inject LoRA layers
- No debug logging to verify successful injection
- Unusual output patterns suggest random/incorrect transformations

**Test**: Add debug logging to verify:
```python
if hasattr(flux_pipeline.transformer, 'peft_config'):
    print(f"PEFT adapters loaded: {list(flux_pipeline.transformer.peft_config.keys())}")
```

### **Secondary Suspect: Incorrect Prompt Format**

The perception adapters may have been trained with different prompts than we're using.

**Evidence**:
- Prompt content can significantly affect adapter behavior
- OmniX paper doesn't specify prompt format
- Current implementation uses long descriptive prompts

**Test**: Try these prompt variations:
1. Empty prompt: `""`
2. Simple task name: `f"{task}"`
3. Minimal: `f"perception: {task}"`

### **Tertiary Suspect: High LoRA Scale**

`lora_scale=1.3` may be too aggressive.

**Evidence**:
- Normal map shows extreme saturation (potential over-application)
- Default is typically 1.0
- Higher scales can amplify errors

**Test**: Reduce to `lora_scale=1.0` or even `0.8` to see if quality improves.

## Recommended Fixes

### Fix 1: Add Comprehensive Debug Logging ✅ IMPLEMENTED

Added logging to verify:
- Adapter weight structure when loading
- Active adapters after `set_adapters()`
- PEFT configuration status

**Code**: `nodes_diffusers.py` lines 360-366, 610-617

### Fix 2: Test Prompt Variations

Create a prompt testing node or modify the perception node to try:

```python
# Option 1: Empty prompt (unconditional)
prompt = ""

# Option 2: Simple task name
prompt = task  # "distance", "normal", etc.

# Option 3: Minimal task descriptor
prompt = f"{task} map"

# Current (potentially wrong):
prompt = f"perception task: {task}, equirectangular projection, 360 degree panoramic view"
```

### Fix 3: Add Adapter Verification

Create diagnostic function to verify PEFT integration:

```python
def verify_adapters_loaded(pipeline, expected_adapters):
    """Verify that LoRA adapters are actually injected into the model."""
    if not hasattr(pipeline.transformer, 'peft_config'):
        logger.error("PEFT not initialized - adapters will not be applied!")
        return False

    loaded = list(pipeline.transformer.peft_config.keys())
    logger.info(f"PEFT adapters in transformer: {loaded}")

    for adapter in expected_adapters:
        if adapter not in loaded:
            logger.warning(f"Adapter '{adapter}' not found in PEFT config!")
            return False

    return True
```

### Fix 4: Update README with Correct lora_scale

Change default recommendation from 1.3 to 1.0:

```markdown
| OmniXLoRALoader | ... | lora_scale | 1.0 (default) | 0.5-1.5 range |
```

### Fix 5: Investigate Weight Key Compatibility

Create diagnostic script to:
1. Load an OmniX adapter
2. Load the Flux model
3. Compare layer names
4. Identify mismatches

**File**: `debug_adapter_weights.py` ✅ CREATED

## Next Steps

1. **User Testing**: Ask user to try `lora_scale=1.0` first (easiest test)
2. **Prompt Testing**: Try empty prompt `""` to see if quality improves
3. **Debug Logging**: Run with logging enabled to verify adapter injection
4. **Key Matching**: If above fails, investigate weight key compatibility
5. **Alternative Approach**: Consider using the weight converter + cross_lora.py path if Diffusers PEFT path proves incompatible

## Technical Notes

### Diffusers PEFT Integration

When `FluxPipeline.load_lora_weights()` is called, it should:
1. Load the safetensors file
2. Parse the weight keys
3. Map keys to model layer names
4. Inject LoRA modules using PEFT
5. Register adapters in `model.peft_config`

If step 3 (key mapping) fails, adapters won't be applied but no error is raised!

### Expected Weight Key Format (OmniX)

Based on Diffusers Flux implementation:
```
transformer.transformer_blocks.{N}.attn.to_q.lora_A.weight
transformer.transformer_blocks.{N}.attn.to_q.lora_B.weight
transformer.transformer_blocks.{N}.attn.to_k.lora_A.weight
transformer.transformer_blocks.{N}.attn.to_k.lora_B.weight
transformer.transformer_blocks.{N}.attn.to_v.lora_A.weight
transformer.transformer_blocks.{N}.attn.to_v.lora_B.weight
transformer.single_transformer_blocks.{N}.attn.to_q.lora_A.weight
...
```

### Expected Model Layer Names (Flux in Diffusers)

Need to verify these match the adapter keys!

## Additional Resources

- OmniX Paper: https://arxiv.org/abs/2510.26800
- OmniX Repository: https://github.com/HKU-MMLab/OmniX
- OmniX Adapters: https://huggingface.co/KevinHuang/OmniX
- Diffusers LoRA docs: https://huggingface.co/docs/diffusers/using-diffusers/loading_adapters

---

**Investigation Status**: Debug logging added, awaiting user testing with:
1. `lora_scale=1.0` (instead of 1.3)
2. Empty prompt `""` (instead of descriptive prompt)
3. Debug output to verify adapter injection
