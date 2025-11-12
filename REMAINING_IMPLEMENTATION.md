# OmniX Perception - Remaining Implementation Steps

**Project**: ComfyUI-OmniX (Perception-Only)
**Status**: VAE Pipeline Working ✅, Denoising Loop Pending ⚠️
**Last Updated**: January 11, 2025
**Version**: 0.3.0-perception

---

## Executive Summary

The ComfyUI-OmniX perception implementation has successfully completed the infrastructure phase. The VAE encoding/decoding pipeline is working correctly, and the node structure is complete.

**What's Working:**
- ✅ All 3 perception nodes load without errors
- ✅ Flux VAE encoding: `panorama → latents` (1024×2048 → 16×128×256)
- ✅ Flux VAE decoding: `latents → image` (16×128×256 → 1024×2048)
- ✅ Adapter loading from `.safetensors` files
- ✅ Proper tensor format handling (ComfyUI format)

**What's Missing:**
- ⚠️ Flux model integration (no access to Flux transformer yet)
- ⚠️ LoRA injection into Flux (framework ready, not activated)
- ⚠️ Denoising loop (the core perception mechanism)
- ⚠️ Conditioning system (RGB guidance for perception)

**Current Behavior:**
The pipeline performs `VAE encode → VAE decode`, returning a reconstruction of the input panorama. Real perception outputs (depth, normals, materials) will only appear after implementing the denoising loop.

---

## Table of Contents

1. [Current Architecture](#current-architecture)
2. [How OmniX Perception Actually Works](#how-omnix-perception-actually-works)
3. [Implementation Roadmap](#implementation-roadmap)
4. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
5. [Technical Reference](#technical-reference)
6. [Testing Strategy](#testing-strategy)
7. [Troubleshooting Guide](#troubleshooting-guide)

---

## Current Architecture

### Working Components

```
┌──────────────────────────────────────────────────────────┐
│ ComfyUI Node Structure (Implemented ✅)                 │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. OmniXAdapterLoader                                  │
│     - Loads .safetensors adapter weights                │
│     - Returns AdapterManager instance                   │
│     - Status: ✅ Working                                 │
│                                                          │
│  2. OmniXPanoramaPerception                             │
│     - Main perception node                              │
│     - Inputs: VAE, adapters, panorama                   │
│     - Outputs: 5 property maps                          │
│     - Status: ⚠️  MVP (just VAE encode/decode)          │
│                                                          │
│  3. OmniXPanoramaValidator                              │
│     - Aspect ratio validation/correction                │
│     - Status: ✅ Working                                 │
│                                                          │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ Current Pipeline Flow                                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Input Panorama (1024×2048×3)                           │
│         ↓                                                │
│  [VAE Encode] ✅                                         │
│         ↓                                                │
│  Latents (16×128×256) ✅                                 │
│         ↓                                                │
│  ❌ MISSING: Denoising with LoRA                        │
│         ↓                                                │
│  [VAE Decode] ✅                                         │
│         ↓                                                │
│  Output = Reconstructed Input                           │
│  (NOT actual perception yet)                            │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Code Status

**File: `nodes.py`**
- `OmniXPerceptionPipeline` class: ✅ Created, ⚠️ Incomplete
  - `encode_to_latents()`: ✅ Working
  - `decode_from_latents()`: ✅ Working
  - `perceive()`: ⚠️ Stub (needs denoising loop)

**File: `omnix/adapters.py`**
- `AdapterManager`: ✅ Loads weights
- `inject_cross_lora_into_model()`: ✅ Framework ready
- Status: Ready for use, not yet called

**File: `omnix/cross_lora.py`**
- `CrossLoRALinear`: ✅ Implemented
- `inject_cross_lora_into_model()`: ✅ Implemented
- `set_active_adapters()`: ✅ Implemented
- Status: Ready for integration

---

## How OmniX Perception Actually Works

### Key Insight: Perception is Conditional Denoising

OmniX does **NOT** perform direct forward inference. Instead, it uses **guided denoising**:

```
Traditional Approach (What We Don't Do):
    RGB Panorama → Neural Network → Depth Map

OmniX Approach (What We Do):
    RGB Panorama → VAE Encode → RGB Latents (conditioning)
                                     ↓
    Noise → Denoising Loop (guided by RGB + LoRA) → Perception Latents
                                     ↓
    Perception Latents → VAE Decode → Depth Map
```

### Why Denoising?

1. **Leverage Flux's Priors**: Flux.1-dev learned rich visual understanding from billions of images. Denoising accesses these priors.

2. **Adapter Efficiency**: Small LoRA adapters (1-2GB) guide the massive Flux model (23GB) toward specific tasks.

3. **High Quality**: Denoising produces better outputs than direct regression.

4. **Multi-task Capability**: Same base model + different adapters = different perception tasks.

### The Full Pipeline (What We Need to Build)

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Encode RGB Panorama to Latents                         │
│                                                                 │
│  Input: RGB panorama (1024×2048×3) in [0,1]                   │
│  ↓                                                              │
│  Flux VAE Encode                                               │
│  ↓                                                              │
│  RGB Latents: (16×128×256) ✅ WORKING                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Load & Inject LoRA Adapter                             │
│                                                                 │
│  - Load task-specific adapter weights (.safetensors)           │
│  - Find Flux transformer blocks                                │
│  - Inject LoRA into attention layers (to_q, to_k, to_v)        │
│  - Set active adapter (e.g., "distance")                       │
│                                                                 │
│  Status: ⚠️  PENDING - Framework ready, needs integration      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Prepare Starting Point (Add Noise)                     │
│                                                                 │
│  RGB Latents + Gaussian Noise → Noisy Latents                 │
│                                                                 │
│  Noise strength: ~0.1 (10% noise)                             │
│  This creates the starting point for denoising                 │
│                                                                 │
│  Status: ⚠️  PENDING                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Prepare Conditioning                                   │
│                                                                 │
│  Create conditioning dict:                                      │
│  {                                                              │
│    'rgb_latents': RGB latents from step 1,                    │
│    'task': 'distance',                                         │
│  }                                                              │
│                                                                 │
│  Flux will use this to guide denoising                         │
│                                                                 │
│  Status: ⚠️  PENDING                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Run Denoising Loop (CORE MECHANISM)                    │
│                                                                 │
│  For step in range(num_steps):  # Default: 28 steps           │
│    1. Get noise prediction from Flux (LoRA active)            │
│    2. Apply conditioning (RGB latents guide prediction)        │
│    3. Update noisy latents → less noisy                        │
│    4. Repeat until clean                                       │
│                                                                 │
│  Input: Noisy latents + RGB conditioning                       │
│  Output: Denoised perception latents                           │
│                                                                 │
│  This is where the magic happens:                              │
│  - LoRA adapter steers Flux toward perception output           │
│  - RGB conditioning provides scene information                 │
│  - Result: Latents representing depth/normals/materials        │
│                                                                 │
│  Status: ⚠️  PENDING - This is the critical missing piece      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: Decode Perception Latents to Image                     │
│                                                                 │
│  Perception Latents (16×128×256)                               │
│  ↓                                                              │
│  Flux VAE Decode                                               │
│  ↓                                                              │
│  Perception Output: (1024×2048×3) ✅ WORKING                   │
│                                                                 │
│  Result: Actual depth/normal/albedo map!                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Roadmap

### Phase 1: Infrastructure ✅ COMPLETE

- [x] Node structure created
- [x] VAE encoding/decoding working
- [x] Adapter loading system
- [x] Cross-LoRA framework
- [x] Import errors fixed
- [x] Nodes load in ComfyUI

**Result**: Foundation is solid, ready for core implementation.

---

### Phase 2: Core Perception ⚠️ IN PROGRESS

**Priority Tasks (In Order):**

#### Task 2.1: Add Flux Model Input ⏳ NEXT
**File**: `nodes.py` → `OmniXPanoramaPerception`

**Change**:
```python
# Add MODEL input to node
"required": {
    "model": ("MODEL", {  # NEW
        "tooltip": "Flux.1-dev model for perception"
    }),
    "vae": ("VAE", {...}),
    ...
}
```

**Why**: Currently the pipeline has no access to Flux transformer. Need this to inject LoRA and run denoising.

**Testing**: Connect Flux model from CheckpointLoader, verify it loads.

---

#### Task 2.2: Implement LoRA Injection 🔄 READY
**File**: `nodes.py` → `OmniXPerceptionPipeline.perceive()`

**Add**:
```python
def perceive(self, panorama, task, num_steps):
    # Encode
    condition_latents = self.encode_to_latents(panorama)

    # ✨ NEW: Inject LoRA
    self.adapter_manager.inject_adapters_into_flux(
        self.flux_model,
        adapter_types=[task]
    )
    self.adapter_manager.set_active_adapter(self.flux_model, task)

    # Continue...
```

**Why**: Adapters must be injected into Flux transformer before denoising.

**Testing**:
- Verify adapters inject without errors
- Check transformer has new LoRA layers
- Ensure memory doesn't explode

---

#### Task 2.3: Implement Noise Addition 📝 PENDING
**File**: `nodes.py` → `OmniXPerceptionPipeline`

**Add new method**:
```python
def add_noise_to_latents(self, latents, noise_strength=0.1):
    """Add Gaussian noise for denoising start"""
    noise = torch.randn_like(latents)
    noisy = latents + noise * noise_strength
    return noisy
```

**Call in `perceive()`**:
```python
noisy_latents = self.add_noise_to_latents(
    condition_latents,
    noise_strength=0.1  # Tune this
)
```

**Why**: Denoising needs a noisy starting point.

**Testing**:
- Check noise is added correctly
- Try noise strengths: 0.05, 0.1, 0.2
- Verify latent statistics

---

#### Task 2.4: Implement Conditioning Preparation 📝 PENDING
**File**: `nodes.py` → `OmniXPerceptionPipeline`

**Add new method**:
```python
def prepare_conditioning(self, condition_latents, task):
    """Prepare RGB conditioning for Flux"""
    # ComfyUI conditioning format
    conditioning = [[
        condition_latents,
        {"task": task}
    ]]
    return conditioning
```

**Why**: Flux needs conditioning in specific format to guide denoising.

**Testing**:
- Verify conditioning structure matches ComfyUI format
- Compare with CLIPTextEncode output
- Check tensor shapes

---

#### Task 2.5: Implement Denoising Loop 🎯 CRITICAL
**File**: `nodes.py` → `OmniXPerceptionPipeline`

**Add new method**:
```python
def run_denoising(
    self,
    flux_model,
    noisy_latents,
    conditioning,
    num_steps=28,
    cfg=1.0
):
    """
    Run Flux denoising with LoRA adapter active.

    This is the core perception mechanism.
    """
    import comfy.sample

    # Prepare latent dict
    latent_dict = {"samples": noisy_latents}

    # Run sampling
    result = comfy.sample.sample(
        model=flux_model,
        noise=noisy_latents,
        steps=num_steps,
        cfg=cfg,
        sampler_name="euler",
        scheduler="normal",
        positive=conditioning,
        negative=None,
        latent_image=latent_dict,
        denoise=1.0
    )

    return result[0]["samples"]
```

**Call in `perceive()`**:
```python
# After noise and conditioning
denoised_latents = self.run_denoising(
    self.flux_model,
    noisy_latents,
    conditioning,
    num_steps=num_steps
)

# Decode
output = self.decode_from_latents(denoised_latents)
```

**Why**: This is where perception actually happens. The denoising loop with LoRA active produces the perception output.

**Testing**:
- Start with 5 steps (fast testing)
- Verify no crashes
- Check output shape
- Gradually increase to 28 steps
- Monitor execution time

---

#### Task 2.6: Parameter Tuning 🔧 OPTIMIZATION
**File**: `nodes.py` → `OmniXPanoramaPerception` INPUT_TYPES

**Add parameters**:
```python
"optional": {
    ...
    "noise_strength": ("FLOAT", {
        "default": 0.1,
        "min": 0.0,
        "max": 1.0,
        "tooltip": "Noise level for denoising start"
    }),
    "cfg": ("FLOAT", {
        "default": 1.0,
        "min": 0.0,
        "max": 10.0,
        "tooltip": "Guidance scale (usually 1.0-2.0 for perception)"
    }),
}
```

**Why**: Allow users to tune perception quality.

**Testing**:
- Try different noise strengths
- Test CFG values
- Find optimal settings for each task

---

### Phase 3: Quality & Polish 📅 FUTURE

- [ ] Compare outputs with official OmniX
- [ ] Optimize performance
- [ ] Add progress bars
- [ ] Improve error messages
- [ ] Remove debug logging
- [ ] Write user documentation
- [ ] Create example workflows

---

## Step-by-Step Implementation Guide

### Step 1: Add Flux Model to Pipeline

**Location**: `nodes.py` line ~195 (`OmniXPanoramaPerception.INPUT_TYPES`)

**Current Code**:
```python
def INPUT_TYPES(cls):
    return {
        "required": {
            "vae": ("VAE", {...}),
            "adapters": ("OMNIX_ADAPTERS", {...}),
            "panorama": ("IMAGE", {...}),
        },
```

**Add MODEL Input**:
```python
def INPUT_TYPES(cls):
    return {
        "required": {
            "model": ("MODEL", {  # ← ADD THIS
                "tooltip": "Flux.1-dev model for perception processing. Connect from CheckpointLoaderSimple or UNETLoader."
            }),
            "vae": ("VAE", {...}),
            "adapters": ("OMNIX_ADAPTERS", {...}),
            "panorama": ("IMAGE", {...}),
        },
```

**Update Function Signature** (line ~240):
```python
# Before
def perceive_panorama(self, vae, adapters, panorama, ...):

# After
def perceive_panorama(self, model, vae, adapters, panorama, ...):  # ← ADD model
```

**Pass to Pipeline** (line ~253):
```python
# Before
pipeline = OmniXPerceptionPipeline(
    vae=vae,
    adapter_manager=adapters,
    ...
)

# After
pipeline = OmniXPerceptionPipeline(
    flux_model=model,  # ← ADD THIS
    vae=vae,
    adapter_manager=adapters,
    ...
)
```

**Update Pipeline Init** (line ~44):
```python
# Before
def __init__(self, vae, adapter_manager, device=None, dtype=torch.bfloat16):
    self.vae = vae
    self.adapter_manager = adapter_manager
    ...

# After
def __init__(self, flux_model, vae, adapter_manager, device=None, dtype=torch.bfloat16):
    self.flux_model = flux_model  # ← ADD THIS
    self.vae = vae
    self.adapter_manager = adapter_manager
    ...
```

**Test**:
1. Reload ComfyUI custom nodes
2. Add OmniXPanoramaPerception node
3. Verify MODEL input appears
4. Connect Flux model from CheckpointLoader
5. Run workflow - should still work (just VAE pass-through)

---

### Step 2: Inject LoRA Adapters

**Location**: `nodes.py` line ~116 (`OmniXPerceptionPipeline.perceive`)

**Current Code**:
```python
def perceive(self, panorama, task, num_steps):
    print(f"[Perception] Running {task} perception with {num_steps} steps")

    # Encode panorama to latents
    condition_latents = self.encode_to_latents(panorama)

    # TODO: Add denoising here

    output = self.decode_from_latents(condition_latents)
    return output
```

**Add LoRA Injection**:
```python
def perceive(self, panorama, task, num_steps):
    print(f"[Perception] Running {task} perception with {num_steps} steps")

    # Encode panorama to latents
    condition_latents = self.encode_to_latents(panorama)

    # ✨ NEW: Inject LoRA adapter into Flux
    print(f"[Perception] Injecting {task} adapter into Flux transformer...")
    self.adapter_manager.inject_adapters_into_flux(
        self.flux_model,
        adapter_types=[task],
        force_reload=False
    )
    self.adapter_manager.set_active_adapter(self.flux_model, task)
    print(f"[Perception] ✓ Adapter '{task}' active in Flux")

    # TODO: Add noise and denoising here

    output = self.decode_from_latents(condition_latents)
    return output
```

**Test**:
1. Run workflow
2. Check console for injection messages
3. Verify no errors
4. Check VRAM usage (should increase slightly)
5. Output still just reconstruction (expected)

---

### Step 3: Add Noise to Latents

**Location**: `nodes.py` after `decode_from_latents` method (line ~114)

**Add New Method**:
```python
def add_noise_to_latents(
    self,
    latents: torch.Tensor,
    noise_strength: float = 0.1
) -> torch.Tensor:
    """
    Add Gaussian noise to latents for denoising start point.

    OmniX perception starts from lightly noised RGB latents,
    then denoises toward the perception output guided by LoRA.

    Args:
        latents: Clean RGB latents from VAE encode
        noise_strength: Amount of noise (0.1 = 10% of latent magnitude)

    Returns:
        Noisy latents as starting point for denoising
    """
    # Generate Gaussian noise same shape as latents
    noise = torch.randn_like(latents)

    # Add scaled noise
    noisy_latents = latents + noise * noise_strength

    print(f"[Perception] Added noise (strength={noise_strength:.3f}) to latents")
    print(f"  Original range: [{latents.min():.3f}, {latents.max():.3f}]")
    print(f"  Noisy range: [{noisy_latents.min():.3f}, {noisy_latents.max():.3f}]")

    return noisy_latents
```

**Use in perceive()** (after LoRA injection):
```python
# After adapter injection
...
print(f"[Perception] ✓ Adapter '{task}' active in Flux")

# ✨ NEW: Add noise
noisy_latents = self.add_noise_to_latents(
    condition_latents,
    noise_strength=0.1  # Start conservative
)

# TODO: Run denoising with noisy_latents

output = self.decode_from_latents(condition_latents)  # Still using clean for now
return output
```

**Test**:
1. Run workflow
2. Check console for noise statistics
3. Verify noisy range is larger than original
4. Try different noise strengths: 0.05, 0.1, 0.2
5. Output still reconstruction (expected)

---

### Step 4: Prepare Conditioning

**Location**: `nodes.py` after `add_noise_to_latents` method

**Add New Method**:
```python
def prepare_conditioning(
    self,
    condition_latents: torch.Tensor,
    task: str
) -> List:
    """
    Prepare conditioning for Flux sampling.

    In OmniX perception, the RGB panorama latents serve as
    conditioning to guide the denoising toward perception output.

    Args:
        condition_latents: Encoded RGB panorama latents
        task: Current perception task (distance, normal, etc.)

    Returns:
        Conditioning in ComfyUI format: [[cond_tensor, extra_dict]]
    """
    # ComfyUI conditioning format
    # Reference: CLIPTextEncode output format
    conditioning = [[
        condition_latents,  # Condition tensor
        {
            "pooled_output": None,  # Flux may need this
            "task": task,           # Track which task
            "guidance": "perception"  # Mark as perception conditioning
        }
    ]]

    print(f"[Perception] Prepared conditioning for task '{task}'")
    print(f"  Condition shape: {condition_latents.shape}")

    return conditioning
```

**Note**: This format may need adjustment. Reference ComfyUI's CLIPTextEncode output to match the exact structure Flux expects.

---

### Step 5: Implement Denoising Loop (CRITICAL)

**Location**: `nodes.py` after `prepare_conditioning` method

**Add New Method**:
```python
def run_denoising(
    self,
    flux_model,
    noisy_latents: torch.Tensor,
    conditioning: List,
    num_steps: int = 28,
    cfg: float = 1.0,
    sampler_name: str = "euler",
    scheduler: str = "normal"
) -> torch.Tensor:
    """
    Run Flux denoising loop with LoRA adapter active.

    This is the CORE perception mechanism. The LoRA adapter
    guides Flux's denoising toward the perception output.

    Args:
        flux_model: Flux model with LoRA injected
        noisy_latents: Starting point (RGB latents + noise)
        conditioning: RGB conditioning from prepare_conditioning()
        num_steps: Denoising steps (OmniX default: 28)
        cfg: Classifier-free guidance scale (low for perception: 1.0-2.0)
        sampler_name: Sampling algorithm
        scheduler: Noise schedule

    Returns:
        Denoised latents (perception output in latent space)
    """
    import comfy.sample
    import comfy.samplers

    print(f"[Perception] Running {num_steps}-step denoising loop...")
    print(f"  Sampler: {sampler_name}")
    print(f"  Scheduler: {scheduler}")
    print(f"  CFG: {cfg}")
    print(f"  Input shape: {noisy_latents.shape}")

    # Prepare latent dict (ComfyUI format)
    latent_dict = {"samples": noisy_latents}

    # Run Flux sampling with LoRA active
    try:
        result = comfy.sample.sample(
            model=flux_model,
            noise=noisy_latents,         # Starting point
            steps=num_steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=conditioning,        # RGB guidance
            negative=None,                # No negative for perception
            latent_image=latent_dict,
            denoise=1.0,                  # Full denoising
        )

        denoised_latents = result[0]["samples"]

        print(f"[Perception] ✓ Denoising complete")
        print(f"  Output shape: {denoised_latents.shape}")

        return denoised_latents

    except Exception as e:
        print(f"[Perception] ✗ Denoising failed: {str(e)}")
        raise RuntimeError(f"Perception denoising failed: {str(e)}")
```

**Integrate into perceive()**:
```python
def perceive(self, panorama, task, num_steps):
    print(f"[Perception] Running {task} perception with {num_steps} steps")

    # 1. Encode
    condition_latents = self.encode_to_latents(panorama)

    # 2. Inject LoRA
    self.adapter_manager.inject_adapters_into_flux(...)
    self.adapter_manager.set_active_adapter(...)

    # 3. Add noise
    noisy_latents = self.add_noise_to_latents(
        condition_latents,
        noise_strength=0.1
    )

    # 4. Prepare conditioning
    conditioning = self.prepare_conditioning(
        condition_latents,
        task=task
    )

    # 5. ✨ NEW: Run denoising
    denoised_latents = self.run_denoising(
        flux_model=self.flux_model,
        noisy_latents=noisy_latents,
        conditioning=conditioning,
        num_steps=num_steps,
        cfg=1.0,
        sampler_name="euler",
        scheduler="normal"
    )

    # 6. Decode
    output = self.decode_from_latents(denoised_latents)

    return output
```

**Test**:
1. Start with `num_steps=5` (very fast)
2. Run workflow
3. Check console for all debug messages
4. Verify no errors
5. Check output - should be different from input now!
6. Try `num_steps=10, 20, 28`
7. Compare output quality at different step counts

---

## Technical Reference

### ComfyUI Sampling API

**Key Module**: `comfy/sample.py`

**Main Function**:
```python
def sample(
    model,           # Flux model
    noise,           # Starting latents
    steps,           # Number of denoising steps
    cfg,             # Classifier-free guidance scale
    sampler_name,    # Algorithm: euler, dpm++_2m, etc.
    scheduler,       # Schedule: normal, karras, exponential
    positive,        # Positive conditioning (list of cond tensors)
    negative,        # Negative conditioning (or None)
    latent_image,    # Dict: {"samples": latent_tensor}
    denoise=1.0,     # Denoise amount (1.0 = full)
    ...
)
```

**Available Samplers**:
- `"euler"`: Fast, good quality (recommended for testing)
- `"dpm++_2m"`: Higher quality, slower
- `"dpmpp_2m_sde"`: Highest quality, slowest
- `"euler_ancestral"`: More variation

**Available Schedulers**:
- `"normal"`: Standard linear schedule
- `"karras"`: Often better quality
- `"exponential"`: Different noise curve
- `"sgm_uniform"`: Flux native (may work best)

### OmniX Parameters

From original implementation analysis:

```python
# Perception defaults
num_steps = 28           # OmniX standard
cfg = 1.0                # Low CFG for perception
noise_strength = 0.1     # Conservative noise level
sampler = "euler"        # Fast and stable
scheduler = "normal"     # Standard
```

### Expected Latent Shapes

```python
# For 1024×2048 panorama:
Input Image:      (1, 1024, 2048, 3)
Encoded Latents:  (1, 16, 128, 256)    # 8x downscale
After Denoising:  (1, 16, 128, 256)    # Same shape
Decoded Output:   (1, 1024, 2048, 3)   # Back to original size
```

### Adapter Injection Points

Flux.1-dev transformer structure (to target with LoRA):
```
flux_model.diffusion_model
    └── transformer_blocks[0..N]
        └── attn
            ├── to_q    ← Inject LoRA here
            ├── to_k    ← Inject LoRA here
            ├── to_v    ← Inject LoRA here
            └── to_out  ← Inject LoRA here
```

### Memory Considerations

```
Component                Memory
----------------------------------
Flux.1-dev model:       ~23GB
Flux VAE:               ~200MB
Single adapter:         ~1.5GB
Working latents:        ~50MB
Total for perception:   ~25GB VRAM

Recommendations:
- 24GB VRAM: All tasks
- 16GB VRAM: Use fp8 Flux + offloading
- 12GB VRAM: Use fp8 + aggressive offloading + lower resolution
- 8GB VRAM:  Perception only (no generation)
```

---

## Testing Strategy

### Phase 1: Integration Testing (Week 1)

**Goal**: Get denoising running without errors

1. **Test Flux Model Loading**
   - Connect MODEL from CheckpointLoader
   - Verify model loads in pipeline
   - Check model structure with debug logging

2. **Test LoRA Injection**
   - Run with single adapter (distance)
   - Check console for injection messages
   - Verify no crashes
   - Monitor VRAM usage

3. **Test Noise Addition**
   - Check noise statistics in console
   - Verify latent ranges
   - Try strengths: 0.05, 0.1, 0.2

4. **Test Basic Denoising**
   - Start with 5 steps (fast)
   - Check for errors
   - Verify output shape correct
   - Output should be different from input

5. **Test Full Pipeline**
   - Run with 28 steps
   - Check execution time (~10-20s expected)
   - Monitor memory usage
   - Verify all 5 tasks run

---

### Phase 2: Quality Testing (Week 2)

**Goal**: Verify perception outputs are meaningful

1. **Visual Inspection**
   - Distance: Should show depth (near=dark, far=bright)
   - Normal: Should show surface orientations (RGB=XYZ)
   - Albedo: Should show base colors
   - Roughness: Should show material roughness
   - Metallic: Should show metallic surfaces

2. **Compare with Official OmniX**
   - Run same panorama through official OmniX
   - Run through ComfyUI-OmniX
   - Visual comparison
   - Compute similarity metrics (MSE, SSIM)

3. **Test Different Panoramas**
   - Indoor scenes
   - Outdoor scenes
   - Different resolutions (512, 1024, 2048)
   - Different aspect ratios (after validator)

4. **Parameter Tuning**
   - Noise strength: 0.05, 0.1, 0.15, 0.2
   - CFG: 0.5, 1.0, 1.5, 2.0
   - Steps: 10, 20, 28, 50
   - Find optimal settings per task

---

### Phase 3: Performance Testing (Week 3)

**Goal**: Optimize for speed and memory

1. **Benchmark Execution**
   - Measure time per task
   - Target: <10s per task at 1024×2048
   - Profile bottlenecks

2. **Memory Profiling**
   - Monitor VRAM throughout pipeline
   - Identify memory leaks
   - Test cleanup after perception

3. **Batch Processing**
   - Test multiple panoramas
   - Check memory scaling
   - Optimize batching

4. **Resolution Scaling**
   - Test 512, 1024, 2048, 4096
   - Measure time vs resolution
   - Find practical limits

---

## Troubleshooting Guide

### Common Issues & Solutions

#### Issue: "Cannot find model"
```
Error: OmniXPerceptionPipeline.__init__() got an unexpected keyword argument 'flux_model'
```

**Cause**: Old code cached in Python
**Solution**: Restart ComfyUI completely (not just reload nodes)

---

#### Issue: "Conditioning format error"
```
Error: Expected list, got torch.Tensor
```

**Cause**: Conditioning format doesn't match Flux expectations
**Solution**: Check CLIPTextEncode output format, match exactly

---

#### Issue: "CUDA out of memory"
```
RuntimeError: CUDA out of memory
```

**Cause**: Not enough VRAM for Flux + adapters
**Solutions**:
1. Use fp8 Flux model
2. Enable model CPU offload
3. Reduce resolution (1024→512)
4. Close other applications

---

#### Issue: "LoRA injection fails"
```
Error: Cannot find transformer blocks
```

**Cause**: Flux model structure different than expected
**Solution**:
1. Add debug logging to print model structure
2. Update injection code to match actual structure
3. Check ComfyUI Flux model implementation

---

#### Issue: "Output is still input reconstruction"
```
Denoising runs but output looks identical to input
```

**Cause**: LoRA not actually affecting denoising
**Debug**:
1. Verify adapter is injected (check console messages)
2. Verify adapter is set as active
3. Check adapter weights are loaded correctly
4. Verify forward hooks are being called
5. Increase noise strength to test if denoising is working

---

#### Issue: "Output is random noise"
```
Denoising produces garbage
```

**Causes & Solutions**:
1. **Too much noise**: Reduce noise_strength to 0.05
2. **Too few steps**: Increase num_steps to 50
3. **Wrong conditioning**: Check conditioning format
4. **Adapter mismatch**: Verify correct adapter for task

---

#### Issue: "Very slow execution"
```
Each perception task takes >60s
```

**Causes & Solutions**:
1. **Model on CPU**: Verify model on GPU
2. **No offloading**: May need some offload if low VRAM
3. **Too many steps**: Try reducing to 20
4. **Large resolution**: Test with 512×1024 first

---

## Next Immediate Actions

### This Session (Right Now)

1. ✅ Document current status (this file)
2. ⏳ User reviews plan
3. ⏳ Decide on next concrete step

### Next Session (Implementation)

1. Add MODEL input to OmniXPanoramaPerception node
2. Update pipeline __init__ to accept Flux model
3. Test that Flux model loads
4. Implement LoRA injection in perceive()
5. Test injection works without errors
6. Implement add_noise_to_latents()
7. Test noise addition
8. Implement prepare_conditioning()
9. Implement run_denoising()
10. Test full pipeline with 5 steps
11. Gradually increase steps to 28
12. Verify outputs are different from input
13. Compare with expected perception outputs

### Week 1 Goals

- [ ] All 5 steps implemented
- [ ] Denoising loop running without errors
- [ ] Outputs are different from inputs
- [ ] Basic quality check passes
- [ ] No memory leaks
- [ ] Execution time reasonable (<30s per task)

---

## Success Metrics

### Minimum Viable Product (MVP)

- [ ] Pipeline runs without crashes
- [ ] Outputs are different from input
- [ ] Distance maps show some depth variation
- [ ] Normal maps show some surface structure
- [ ] Execution completes in reasonable time (<1 min total)

### Production Ready

- [ ] Outputs match official OmniX quality (visual comparison)
- [ ] SSIM > 0.8 compared to official OmniX
- [ ] Execution time < 10s per task at 1024×2048
- [ ] Memory usage < 20GB VRAM total
- [ ] Works on 16GB VRAM with fp8
- [ ] All 5 perception tasks functional
- [ ] Parameter tuning options available
- [ ] Error handling robust
- [ ] Documentation complete

---

## Conclusion

The foundation is solid. The VAE pipeline works, adapters load, and the LoRA framework is ready. The remaining work is focused and concrete: integrate Flux model, inject LoRA, and implement the denoising loop.

**Critical Path**:
1. Add Flux model input (10 mins)
2. Inject LoRA (15 mins)
3. Add noise (10 mins)
4. Prepare conditioning (15 mins)
5. Implement denoising loop (30 mins)
6. Test and debug (1-2 hours)

**Total Estimated Time**: 2-3 hours of focused implementation.

**Key Risk**: Conditioning format may need adjustment based on Flux's expectations. This is the most uncertain part and may require iteration.

**Mitigation**: Reference ComfyUI's CLIPTextEncode carefully, test with minimal example first, add extensive debug logging.

---

**Document Status**: Complete and Ready for Implementation
**Next Step**: Review with user, then begin implementation
**Author**: Claude (AI Assistant)
**Project**: ComfyUI-OmniX Perception
