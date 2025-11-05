# ComfyUI-OmniX: Claude Code Agent Guidelines

**Project:** ComfyUI-OmniX Custom Node Implementation  
**Purpose:** Guide Claude Code in implementing OmniX panorama generation and perception for ComfyUI  
**Last Updated:** November 4, 2025

---

## Project Overview

You are implementing a ComfyUI custom node package that integrates OmniX capabilities:
- **Panorama Generation:** Text/image → 360° equirectangular panoramas
- **Panorama Perception:** Extract depth, normals, albedo, roughness, metallic from panoramas
- **Base Model:** Flux.1-dev with custom OmniX adapters
- **Target Users:** ComfyUI users creating 3D environments, VR content, and PBR materials

**Key Documents:**
- Design Doc: `DESIGN_DOC.md`
- Original OmniX: https://github.com/HKU-MMLab/OmniX
- ComfyUI Docs: https://github.com/comfyanonymous/ComfyUI

---

## Project Structure

```
ComfyUI-OmniX/
├── __init__.py                 # Node registration
├── nodes.py                    # All node class definitions
├── omnix/                      # Core implementation
│   ├── __init__.py
│   ├── model_loader.py         # Model weight management
│   ├── generator.py            # Panorama generation logic
│   ├── perceiver.py            # Perception inference
│   ├── adapters.py             # Adapter implementations
│   └── utils.py                # Helper functions
├── tests/                      # Unit and integration tests
│   ├── test_loader.py
│   ├── test_generator.py
│   └── test_perceiver.py
├── workflows/                  # Example ComfyUI workflows (JSON)
├── requirements.txt
├── README.md
└── DESIGN_DOC.md
```

---

## Core Architecture Principles

### 1. ComfyUI Node Patterns

**Every node MUST follow this structure:**

```python
class NodeName:
    """Brief description of what this node does"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "param_name": ("TYPE", {"default": value, "min": min, "max": max}),
            },
            "optional": {
                "optional_param": ("TYPE",),
            }
        }
    
    RETURN_TYPES = ("TYPE1", "TYPE2")
    RETURN_NAMES = ("output1", "output2")  # Optional but recommended
    FUNCTION = "method_name"
    CATEGORY = "OmniX"
    
    def method_name(self, param_name, optional_param=None):
        # Implementation
        return (output1, output2)
```

**Critical Rules:**
- `INPUT_TYPES` must be a classmethod returning a dict
- All inputs in "required" must have defaults or be provided
- `RETURN_TYPES` must match the number of returned values
- `FUNCTION` must match an actual method name
- Always use `CATEGORY = "OmniX"` for organization

### 2. Custom Type System

**Define custom types for type safety:**

```python
# In __init__.py or nodes.py
class OmniXModel:
    """Wrapper for OmniX model components"""
    def __init__(self, base_model, adapters, config):
        self.base_model = base_model
        self.adapters = adapters
        self.config = config
        self.device = base_model.device
    
    def to(self, device):
        self.base_model.to(device)
        for adapter in self.adapters.values():
            adapter.to(device)
        return self

# Register the custom type
NODE_CLASS_MAPPINGS = {
    "OMNIX_MODEL": OmniXModel,
}
```

### 3. Memory Management

**Always implement proper VRAM management:**

```python
import torch
import gc

def cleanup_model(model):
    """Properly cleanup model to free VRAM"""
    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()

class OmniXModelLoader:
    # Class-level cache
    _model_cache = {}
    
    def load_model(self, model_name, precision, offload_mode):
        cache_key = f"{model_name}_{precision}_{offload_mode}"
        
        # Check cache first
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        # Load model
        model = self._load_from_disk(model_name, precision)
        
        # Apply offloading
        if offload_mode == "sequential":
            model.enable_sequential_cpu_offload()
        elif offload_mode == "model":
            model.enable_model_cpu_offload()
        
        # Cache and return
        self._model_cache[cache_key] = model
        return model
```

### 4. Error Handling Patterns

**Implement graceful error handling with user-friendly messages:**

```python
class OmniXPanoramaGenerator:
    def generate(self, omnix_model, prompt, seed, steps, 
                 guidance_scale, width, height, **kwargs):
        try:
            # Validate inputs
            self._validate_inputs(width, height, steps, guidance_scale)
            
            # Set seed for reproducibility
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            # Generation logic
            panorama = self._generate_internal(
                omnix_model, prompt, steps, 
                guidance_scale, width, height
            )
            
            return (panorama,)
            
        except torch.cuda.OutOfMemoryError:
            cleanup_model(omnix_model.base_model)
            raise RuntimeError(
                f"Out of VRAM generating {width}x{height} panorama. "
                f"Try: 1) Lower resolution, 2) Enable model offloading, "
                f"3) Close other applications"
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Model weights not found: {e}. "
                f"Please run the model loader first or check installation."
            )
        except Exception as e:
            raise RuntimeError(
                f"Panorama generation failed: {str(e)}. "
                f"Check ComfyUI console for details."
            )
    
    def _validate_inputs(self, width, height, steps, guidance_scale):
        """Validate input parameters"""
        aspect_ratio = width / height
        if not (1.8 < aspect_ratio < 2.2):
            raise ValueError(
                f"Invalid panorama dimensions {width}x{height}. "
                f"Aspect ratio should be ~2:1 for equirectangular format."
            )
        
        if steps < 10:
            print(f"Warning: {steps} steps is very low, quality may suffer")
        
        if guidance_scale > 10:
            print(f"Warning: High guidance scale {guidance_scale} may oversaturate")
```

---

## Implementation Guidelines

### Model Loading

**When implementing model_loader.py:**

1. **Check multiple locations for weights:**
```python
def find_model_path(model_name):
    """Search for model weights in standard locations"""
    search_paths = [
        Path("models/omnix") / model_name,
        Path("models/diffusion_models") / model_name,
        Path.home() / ".cache/huggingface/hub" / model_name,
    ]
    
    for path in search_paths:
        if path.exists():
            return path
    
    return None
```

2. **Implement lazy loading for adapters:**
```python
class AdapterManager:
    def __init__(self, adapter_dir):
        self.adapter_dir = adapter_dir
        self._loaded_adapters = {}
    
    def get_adapter(self, adapter_name):
        """Load adapter on-demand"""
        if adapter_name not in self._loaded_adapters:
            adapter_path = self.adapter_dir / f"{adapter_name}_adapter.safetensors"
            self._loaded_adapters[adapter_name] = load_safetensors(adapter_path)
        
        return self._loaded_adapters[adapter_name]
    
    def unload_adapter(self, adapter_name):
        """Free memory when adapter not needed"""
        if adapter_name in self._loaded_adapters:
            del self._loaded_adapters[adapter_name]
            gc.collect()
```

3. **Support multiple precision formats:**
```python
def load_with_precision(model_path, precision):
    """Load model with specified precision"""
    if precision == "fp32":
        dtype = torch.float32
    elif precision == "fp16":
        dtype = torch.float16
    elif precision == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unknown precision: {precision}")
    
    model = load_model(model_path)
    return model.to(dtype=dtype)
```

### Panorama Generation

**When implementing generator.py:**

1. **Follow Flux.1-dev inference pattern:**
```python
from diffusers import FluxPipeline
import torch

class OmniXGenerator:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.pipeline = self._create_pipeline()
    
    def _create_pipeline(self):
        """Initialize Flux pipeline with OmniX adapters"""
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16
        )
        
        # Inject OmniX adapters
        self._inject_adapters(pipe)
        
        return pipe.to(self.device)
    
    def _inject_adapters(self, pipeline):
        """Integrate OmniX adapters into Flux pipeline"""
        # This is where OmniX-specific adapter injection happens
        # Based on the cross-modal adapter architecture
        pass
    
    def generate(self, prompt, height, width, num_steps, guidance_scale, seed):
        """Generate panorama"""
        generator = torch.Generator(self.device).manual_seed(seed)
        
        output = self.pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        return output.images[0]
```

2. **Handle image conditioning:**
```python
def generate_with_conditioning(self, prompt, conditioning_image, **kwargs):
    """Generate panorama with image conditioning"""
    if conditioning_image is not None:
        # Encode conditioning image
        cond_latents = self.pipeline.vae.encode(
            conditioning_image
        ).latent_dist.sample()
        
        # Mix with text conditioning
        output = self.pipeline(
            prompt=prompt,
            image=cond_latents,  # Pass as initial latents
            strength=0.8,  # Control conditioning strength
            **kwargs
        )
    else:
        output = self.pipeline(prompt=prompt, **kwargs)
    
    return output.images[0]
```

3. **Convert outputs to ComfyUI format:**
```python
def to_comfyui_image(pil_image):
    """Convert PIL Image to ComfyUI tensor format"""
    import numpy as np
    
    # PIL Image -> NumPy array (H, W, C)
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    
    # Add batch dimension (B, H, W, C)
    image_tensor = torch.from_numpy(image_np)[None,]
    
    return image_tensor

def from_comfyui_image(tensor):
    """Convert ComfyUI tensor to PIL Image"""
    from PIL import Image
    
    # Remove batch dimension and convert to numpy
    image_np = tensor[0].cpu().numpy()
    
    # Scale to 0-255 and convert to uint8
    image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
    
    return Image.fromarray(image_np)
```

### Panorama Perception

**When implementing perceiver.py:**

1. **Run adapters efficiently:**
```python
class OmniXPerceiver:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
    
    def perceive(self, panorama, extract_modes):
        """Extract multiple properties from panorama"""
        # Encode panorama once
        with torch.no_grad():
            encoded = self.model.encode_image(panorama)
        
        outputs = {}
        
        # Run requested adapters
        for mode in extract_modes:
            if mode == "distance":
                outputs["distance"] = self._extract_distance(encoded)
            elif mode == "normal":
                outputs["normal"] = self._extract_normal(encoded)
            elif mode == "albedo":
                outputs["albedo"] = self._extract_albedo(encoded)
            # ... etc
        
        return outputs
    
    def _extract_distance(self, encoded):
        """Run distance adapter"""
        with torch.no_grad():
            distance = self.model.adapters["distance"](encoded)
        
        # Post-process: normalize, convert to metric scale
        distance = self._normalize_distance(distance)
        return distance
```

2. **Handle multi-modal outputs:**
```python
def format_outputs_for_comfyui(outputs):
    """Convert perception outputs to ComfyUI format"""
    formatted = {}
    
    for key, value in outputs.items():
        if key == "distance":
            # Distance: single channel, visualize as heatmap
            formatted[key] = visualize_depth(value)
        elif key == "normal":
            # Normals: 3 channels, map from [-1,1] to [0,1]
            formatted[key] = (value + 1.0) / 2.0
        elif key in ["albedo", "roughness", "metallic"]:
            # Material properties: already in [0,1]
            formatted[key] = value
    
    return formatted

def visualize_depth(depth_map):
    """Create visual representation of depth map"""
    # Normalize to 0-1 range
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    # Apply colormap for better visualization
    import matplotlib.cm as cm
    colormap = cm.get_cmap('viridis')
    depth_colored = colormap(depth_norm.cpu().numpy())[:, :, :3]
    
    return torch.from_numpy(depth_colored).float()
```

---

## Testing Strategy

### Unit Tests

**Create focused unit tests for each component:**

```python
# tests/test_model_loader.py
import pytest
import torch
from omnix.model_loader import OmniXModelLoader

def test_model_loader_initialization():
    """Test model loader can initialize"""
    loader = OmniXModelLoader()
    assert loader is not None

def test_model_loading_fp16():
    """Test loading model in FP16 precision"""
    loader = OmniXModelLoader()
    model = loader.load_model("omnix-base", precision="fp16", offload_mode="none")
    
    assert model is not None
    assert next(model.parameters()).dtype == torch.float16

def test_model_caching():
    """Test that models are cached properly"""
    loader = OmniXModelLoader()
    model1 = loader.load_model("omnix-base", "fp16", "none")
    model2 = loader.load_model("omnix-base", "fp16", "none")
    
    # Should be same instance
    assert model1 is model2

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_loading():
    """Test model loads to GPU when available"""
    loader = OmniXModelLoader()
    model = loader.load_model("omnix-base", "fp16", "none")
    
    assert model.device.type == "cuda"
```

### Integration Tests

**Test complete workflows:**

```python
# tests/test_integration.py
def test_text_to_panorama_workflow():
    """Test complete text-to-panorama generation"""
    # Setup
    loader_node = OmniXModelLoader()
    generator_node = OmniXPanoramaGenerator()
    
    # Load model
    model = loader_node.load_model("omnix-base", "fp16", "model")
    
    # Generate panorama
    panorama = generator_node.generate(
        omnix_model=model,
        prompt="A photorealistic modern living room",
        seed=42,
        steps=28,
        guidance_scale=5.0,
        width=2048,
        height=1024
    )
    
    # Validate output
    assert panorama is not None
    assert panorama.shape == (1, 1024, 2048, 3)
    assert 0 <= panorama.min() <= panorama.max() <= 1

def test_panorama_perception_workflow():
    """Test perception extraction"""
    # Setup
    loader_node = OmniXModelLoader()
    perceiver_node = OmniXPanoramaPerception()
    
    # Load model
    model = loader_node.load_model("omnix-base", "fp16", "model")
    
    # Create test panorama
    test_panorama = torch.rand(1, 512, 1024, 3)
    
    # Extract properties
    distance, normal, albedo, roughness, metallic, _ = perceiver_node.perceive(
        omnix_model=model,
        panorama=test_panorama,
        extract_distance=True,
        extract_normal=True,
        extract_albedo=True,
        extract_roughness=True,
        extract_metallic=True,
        extract_semantic=False
    )
    
    # Validate outputs
    assert distance.shape == (1, 512, 1024, 1)
    assert normal.shape == (1, 512, 1024, 3)
    assert albedo.shape == (1, 512, 1024, 3)
```

---

## Common Tasks & Solutions

### Task 1: Adding a New Node

**Steps:**
1. Define node class in `nodes.py`
2. Implement `INPUT_TYPES`, `RETURN_TYPES`, and main function
3. Add to `NODE_CLASS_MAPPINGS` in `__init__.py`
4. Write tests
5. Update documentation

**Template:**
```python
# In nodes.py
class OmniXNewNode:
    """Description of what this node does"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input1": ("TYPE",),
            }
        }
    
    RETURN_TYPES = ("OUTPUT_TYPE",)
    FUNCTION = "process"
    CATEGORY = "OmniX"
    
    def process(self, input1):
        result = do_something(input1)
        return (result,)

# In __init__.py
NODE_CLASS_MAPPINGS = {
    "OmniXNewNode": OmniXNewNode,
    # ... other nodes
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniXNewNode": "OmniX New Node",
}
```

### Task 2: Debugging VRAM Issues

**Diagnostic approach:**
```python
def diagnose_memory():
    """Print memory usage information"""
    import torch
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    else:
        print("CUDA not available")

# Add to critical sections
def generate(self, ...):
    diagnose_memory()
    # ... generation code
    diagnose_memory()
```

**Solutions:**
1. Enable model offloading
2. Reduce batch size
3. Lower precision (FP16)
4. Clear cache between operations
5. Use gradient checkpointing

### Task 3: Optimizing Inference Speed

**Techniques:**
```python
# 1. Use torch.compile (PyTorch 2.0+)
model = torch.compile(model, mode="reduce-overhead")

# 2. Enable Flash Attention
from torch.nn.attention import SDPBackend, sdpa_kernel
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    output = model(input)

# 3. Use mixed precision
from torch.cuda.amp import autocast
with autocast(dtype=torch.float16):
    output = model(input)

# 4. Batch operations when possible
def process_batch(images):
    # Process all at once instead of loop
    return model(torch.stack(images))
```

### Task 4: Handling Model Updates

**When OmniX releases new weights:**

```python
# In model_loader.py
SUPPORTED_VERSIONS = {
    "omnix-base-v1.0": {
        "url": "https://huggingface.co/...",
        "sha256": "abc123...",
    },
    "omnix-base-v1.1": {
        "url": "https://huggingface.co/...",
        "sha256": "def456...",
    },
}

def load_model(model_name, version="latest"):
    if version == "latest":
        version = max(SUPPORTED_VERSIONS.keys())
    
    config = SUPPORTED_VERSIONS[version]
    model_path = download_if_needed(config["url"], config["sha256"])
    return load_from_path(model_path)
```

---

## Code Quality Standards

### Style Guide

**Follow these conventions:**

```python
# 1. Use type hints
def generate_panorama(
    prompt: str,
    width: int,
    height: int,
    seed: int = 0
) -> torch.Tensor:
    pass

# 2. Document functions
def process_image(image: torch.Tensor) -> torch.Tensor:
    """
    Process input image for OmniX.
    
    Args:
        image: Input tensor in ComfyUI format (B, H, W, C)
    
    Returns:
        Processed tensor ready for model input
    
    Raises:
        ValueError: If image dimensions invalid
    """
    pass

# 3. Use descriptive variable names
# Bad
x = model(i)

# Good
panorama = generator.generate(input_image)

# 4. Keep functions focused
# Each function should do one thing well
def load_model():  # Only loads
    pass

def configure_model():  # Only configures
    pass
```

### Performance Best Practices

```python
# 1. Avoid unnecessary tensor copies
# Bad
result = tensor.clone().detach().cpu()

# Good (if you need all of these)
result = tensor.detach().cpu()

# 2. Use in-place operations when safe
# Bad
tensor = tensor + 1

# Good
tensor += 1

# 3. Minimize CPU-GPU transfers
# Bad
for i in range(len(batch)):
    result = model(batch[i].cuda()).cpu()

# Good
results = model(batch.cuda()).cpu()

# 4. Use context managers for no_grad
with torch.no_grad():
    output = model(input)
```

---

## Troubleshooting Guide

### Problem: "Model weights not found"

**Solution:**
```python
# Check these locations
locations = [
    "ComfyUI/models/omnix/omnix-base",
    "~/.cache/huggingface/hub/omnix",
    "ComfyUI/custom_nodes/ComfyUI-OmniX/models"
]

# Verify file structure
expected_files = [
    "config.json",
    "rgb_adapter.safetensors",
    "distance_adapter.safetensors",
    # ... etc
]
```

### Problem: "CUDA out of memory"

**Solution hierarchy:**
```python
def handle_oom():
    """Try progressively more aggressive memory saving"""
    try:
        return generate_normal()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        try:
            return generate_with_offload()
        except torch.cuda.OutOfMemoryError:
            try:
                return generate_lower_resolution()
            except torch.cuda.OutOfMemoryError:
                return generate_cpu()
```

### Problem: "Invalid panorama aspect ratio"

**Solution:**
```python
def fix_aspect_ratio(image: torch.Tensor, target_ratio: float = 2.0):
    """Crop or pad image to correct aspect ratio"""
    h, w = image.shape[-2:]
    current_ratio = w / h
    
    if abs(current_ratio - target_ratio) < 0.1:
        return image  # Close enough
    
    if current_ratio > target_ratio:
        # Too wide, crop width
        new_w = int(h * target_ratio)
        start = (w - new_w) // 2
        return image[..., :, start:start+new_w, :]
    else:
        # Too narrow, pad width
        new_w = int(h * target_ratio)
        pad = (new_w - w) // 2
        return torch.nn.functional.pad(
            image, (pad, pad, 0, 0), mode='replicate'
        )
```

---

## Integration Checklist

Before considering implementation complete:

### Core Functionality
- [ ] Model loader working with all precision modes
- [ ] Panorama generation from text working
- [ ] Panorama generation from image+text working
- [ ] All perception modalities extracting correctly
- [ ] Outputs in correct ComfyUI format

### Error Handling
- [ ] OOM errors handled gracefully
- [ ] Missing model weights detected with helpful message
- [ ] Invalid inputs validated before processing
- [ ] All exceptions caught and converted to user-friendly errors

### Performance
- [ ] Generation time < 30s on RTX 4090
- [ ] Memory usage < 16GB with offloading
- [ ] Model caching working to avoid reloads
- [ ] Batch processing supported

### Documentation
- [ ] README with installation instructions
- [ ] Node usage examples
- [ ] Example workflows (JSON files)
- [ ] API documentation
- [ ] Troubleshooting guide

### Testing
- [ ] Unit tests for all core components
- [ ] Integration tests for complete workflows
- [ ] Performance benchmarks documented
- [ ] Tested on multiple GPU types

### Polish
- [ ] Proper logging (not excessive print statements)
- [ ] Progress bars for long operations
- [ ] Sensible default values
- [ ] Clear node organization in ComfyUI menu

---

## Communication with User

### When to ask for clarification:

1. **Ambiguous requirements:** "Should the semantic segmentation output be one-hot encoded or label indices?"
2. **Missing information:** "I don't see the OmniX model weights. Do you have a download link?"
3. **Design decisions:** "Should we prioritize memory efficiency or speed for the default settings?"
4. **Architecture choices:** "Should each adapter be a separate node or combined in one node with toggles?"

### How to present solutions:

**Good:**
```
I implemented the panorama generator with three key features:

1. Text-to-panorama generation (28 steps, ~25s on RTX 4090)
2. Image conditioning support for guided generation
3. Automatic model offloading for 8GB+ VRAM

The node is working and I've added tests. Ready for the next component.
```

**Bad:**
```
I added some code for the panorama thing. It might work but I'm not sure about the memory stuff. Let me know if you want me to change anything.
```

---

## Quick Reference

### Essential Imports
```python
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
from PIL import Image
import safetensors
```

### ComfyUI Type Mappings
```python
"INT" → int
"FLOAT" → float
"STRING" → str
"BOOLEAN" → bool
"IMAGE" → torch.Tensor  # (B, H, W, C), [0, 1]
"LATENT" → Dict[str, torch.Tensor]
"MODEL" → Custom model class
```

### Common Tensor Operations
```python
# ComfyUI image format: (batch, height, width, channels)
image = torch.randn(1, 512, 1024, 3)

# Convert to PyTorch format (B, C, H, W)
image_torch = image.permute(0, 3, 1, 2)

# Back to ComfyUI format
image_comfy = image_torch.permute(0, 2, 3, 1)

# Normalize to [0, 1]
image_norm = (image - image.min()) / (image.max() - image.min())
```

---

## Final Notes

**Remember:**
- Always test on both CPU and GPU
- Document WHY, not just WHAT
- Keep nodes simple and composable
- Optimize after it works, not before
- User experience > perfect code

**When stuck:**
1. Check the design doc
2. Look at similar ComfyUI nodes
3. Review OmniX original implementation
4. Ask the user for clarification

**Success metrics:**
- Code works reliably
- Users can understand how to use it
- Performance meets targets
- No critical bugs

Good luck! 🚀
