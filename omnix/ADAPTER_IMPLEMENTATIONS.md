# Adapter Implementations

This directory contains multiple adapter implementation files. This document explains their purpose and relationships.

## Current Architecture (v1.1.0)

### Production Files

#### `adapters.py` ✅ **ACTIVE**
**Purpose:** Current production adapter manager for Diffusers-based implementation

**Key Classes:**
- `AdapterManager`: Manages loading and injection of OmniX LoRA adapters into Flux models
- `OmniXAdapters`: Legacy compatibility wrapper

**Features:**
- Loads adapter weights from .safetensors files
- Injects adapters as LoRA into Flux transformer blocks via `cross_lora.py`
- Manages adapter lifecycle (loading, injecting, activating, removing)
- Supports multiple adapter types: distance, normal, albedo, pbr, semantic

**Used By:**
- `nodes_diffusers.py` (OmniXLoRALoader node)
- Diffusers-based perception workflow

---

#### `cross_lora.py` ✅ **ACTIVE**
**Purpose:** Implements Cross-LoRA system for multi-adapter support

**Key Classes:**
- `CrossLoRALinear`: Linear layer with multiple LoRA adapters that can be dynamically switched

**Key Functions:**
- `inject_cross_lora_into_model()`: Injects LoRA adapters into Flux transformer blocks
- `set_active_adapters()`: Activates specific adapter for inference
- `remove_cross_lora_from_model()`: Removes all adapters and restores original layers

**Architecture:**
- Allows multiple task-specific adapters in one model
- Dynamic adapter switching per forward pass
- Targets Flux's double_blocks (img_attn) and single_blocks (linear1/linear2)

---

### Legacy/Alternative Files

#### `adapters_old.py` 🔄 **LEGACY - KEPT FOR COMPATIBILITY**
**Purpose:** Original standalone adapter module implementation

**Key Classes:**
- `AdapterModule`: Standalone NN module for task-specific adapters

**Why Kept:**
- `adapters.py` imports it for backward compatibility: `from .adapters_old import AdapterModule as LegacyAdapterModule`
- Used by `OmniXPerceiver` class in `perceiver.py`
- Some legacy code paths still reference this

**Status:**
- Not used by Diffusers-based nodes
- Can be removed if `perceiver.py` is confirmed unused

---

#### `adapters_new.py` 🚧 **EXPERIMENTAL - UNUSED**
**Purpose:** Experimental alternative adapter implementation

**Status:**
- Not imported by any production code
- Appears to be a development experiment
- **Candidate for removal** unless documented otherwise

---

## Migration History

### v1.0.0 → v1.1.0 (Diffusers Migration)

**Problem:**
- Original implementation used ComfyUI's native Flux
- Weight conversion caused architectural incompatibilities
- Adapters trained with Diffusers couldn't be properly converted to ComfyUI format

**Solution:**
- Switched to HuggingFace Diffusers' FluxPipeline entirely
- Created new `adapters.py` and `cross_lora.py` for LoRA injection
- Bypassed weight conversion completely

**Result:**
- Perfect compatibility with OmniX adapters
- No weight conversion needed
- Simplified architecture

---

## File Relationships

```
Production Flow (Diffusers):
==========================
nodes_diffusers.py
    └─> adapters.py (AdapterManager)
            └─> cross_lora.py (inject_cross_lora_into_model)
                    └─> weight_converter.py (convert OmniX → ComfyUI format for injection)

Legacy Flow (Unused):
====================
perceiver.py (OmniXPerceiver)
    └─> adapters.py (OmniXAdapters wrapper)
            └─> adapters_old.py (LegacyAdapterModule)

Experimental (Unused):
====================
adapters_new.py (not imported anywhere)
```

---

## Adapter File Formats

### OmniX Official Format
Downloaded from HuggingFace:
- `rgb_to_depth_depth.safetensors`
- `rgb_to_normal_normal.safetensors`
- `rgb_to_albedo_albedo.safetensors`
- `rgb_to_pbr_pbr.safetensors`
- `rgb_to_semantic_semantic.safetensors`

### Key Mapping
```python
ADAPTER_FILENAMES = {
    "distance": "rgb_to_depth_depth.safetensors",
    "normal": "rgb_to_normal_normal.safetensors",
    "albedo": "rgb_to_albedo_albedo.safetensors",
    "pbr": "rgb_to_pbr_pbr.safetensors",
    "semantic": "rgb_to_semantic_semantic.safetensors",
}
```

---

## Recommendations

### Short Term
1. ✅ Keep `adapters.py` and `cross_lora.py` as production files
2. ✅ Keep `adapters_old.py` for compatibility with `perceiver.py`
3. ⚠️  Add clear deprecation warnings to `adapters_old.py`

### Long Term
1. 🔍 Determine if `perceiver.py` (OmniXPerceiver) is still needed
2. 🗑️ Remove `adapters_old.py` if perceiver is unused
3. 🗑️ Remove `adapters_new.py` unless documented as experimental feature
4. 📝 Update all imports to use `adapters.py` directly

---

## Developer Guide

### To Add a New Adapter Type

1. **Update `ADAPTER_FILENAMES` in `adapters.py`:**
   ```python
   ADAPTER_FILENAMES = {
       "your_task": "rgb_to_task_task.safetensors",
   }
   ```

2. **Update `ADAPTER_CONFIGS` in `adapters.py`:**
   ```python
   ADAPTER_CONFIGS = {
       "your_task": {
           "rank": 64,
           "scale": 1.0,
           "targets": ["to_q", "to_k", "to_v"],
       },
   }
   ```

3. **Add to `OmniXLoRALoader` node in `nodes_diffusers.py`:**
   ```python
   ADAPTER_FILENAMES = {
       "your_task": "rgb_to_task_task.safetensors",
   }
   ```

4. **Download weights** and place in `models/loras/omnix/`

### To Use Adapters in Code

```python
from omnix.adapters import AdapterManager

# Initialize
manager = AdapterManager(
    adapter_dir="/path/to/adapters",
    device=torch.device("cuda"),
    dtype=torch.float16
)

# Load specific adapter
weights = manager.load_adapter("distance")

# Inject into Flux model
manager.inject_adapters_into_flux(
    flux_model=model,
    adapter_types=["distance", "normal"]
)

# Set active adapter
manager.set_active_adapter(model, "distance")

# Run inference...

# Cleanup
manager.remove_adapters(model)
```

---

**Last Updated:** 2024 (v1.1.0 - Diffusers-only)
