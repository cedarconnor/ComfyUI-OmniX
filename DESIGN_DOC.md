# OmniX ComfyUI Integration - Design Document

**Project:** ComfyUI-OmniX Custom Nodes  
**Version:** 1.0  
**Date:** November 4, 2025  
**Status:** Design Phase  

---

## Executive Summary

This document outlines the design and implementation strategy for integrating OmniX panorama generation and perception capabilities into ComfyUI as custom nodes. OmniX is a Flux.1-dev based framework that generates 360° panoramic images and extracts comprehensive scene properties including geometry (depth, normals) and PBR materials (albedo, roughness, metallic).

### Key Objectives
1. Enable panorama generation from text and image prompts
2. Extract multi-modal scene properties from panoramas
3. Maintain ComfyUI's node-based workflow paradigm
4. Ensure efficient memory usage and performance
5. Support batch processing and iterative workflows

---

## 1. System Architecture

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ComfyUI Frontend                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              ComfyUI-OmniX Custom Nodes                      │
│  ┌─────────────────┐  ┌──────────────────┐                 │
│  │ OmniX Generator │  │ OmniX Perception │                 │
│  │     Node        │  │      Node        │                 │
│  └────────┬────────┘  └────────┬─────────┘                 │
│           │                     │                            │
│  ┌────────▼─────────────────────▼─────────┐                │
│  │      OmniX Core Adapter Layer          │                │
│  └────────┬───────────────────┬───────────┘                │
└───────────┼───────────────────┼────────────────────────────┘
            │                   │
┌───────────▼───────────┐  ┌───▼────────────────────┐
│  Flux.1-dev Model     │  │  Cross-Modal Adapters  │
│  (Pre-trained)        │  │  (OmniX Components)    │
└───────────────────────┘  └────────────────────────┘
```

### 1.2 Component Breakdown

#### Core Components
1. **OmniX Loader Node**: Loads and manages Flux.1-dev + OmniX adapters
2. **Panorama Generator Node**: Text/image → 360° panorama
3. **Panorama Perception Node**: Panorama → multi-modal outputs
4. **Panorama All-in-One Node**: Combined generation + perception
5. **Utility Nodes**: Format converters, preview, save

#### Supporting Infrastructure
- Model weight management system
- Adapter loading and caching
- Memory optimization layer
- Output format handlers

---

## 2. Technical Specifications

### 2.1 Model Architecture

**Base Model:** Flux.1-dev (Flow Matching Diffusion Transformer)
- **Architecture:** DiT (Diffusion Transformer)
- **Parameters:** ~12B (Flux base)
- **Input:** Text embeddings + optional image conditioning
- **Output:** Equirectangular panorama (512×1024 to 2048×4096)

**OmniX Adapters:** Cross-modal adapter structure
- **Adapter Types:** 
  - RGB generation adapter
  - Distance prediction adapter
  - Normal prediction adapter
  - Albedo extraction adapter
  - Roughness estimation adapter
  - Metallic estimation adapter
- **Design:** Separate adapters (empirically superior to shared adapters)

### 2.2 Input/Output Specifications

#### Panorama Generation Inputs
```python
{
    "prompt": str,                    # Text description
    "image": Optional[torch.Tensor],  # Conditioning image (H×W×3)
    "seed": int,                      # Random seed
    "steps": int,                     # Inference steps (20-50)
    "guidance_scale": float,          # CFG scale (3.5-7.5)
    "width": int,                     # Output width (1024, 2048, 4096)
    "height": int,                    # Output height (512, 1024, 2048)
}
```

#### Panorama Perception Inputs
```python
{
    "panorama": torch.Tensor,         # RGB panorama (H×W×3)
    "output_modes": List[str],        # ["distance", "normal", "albedo", ...]
    "precision": str,                 # "fp32" | "fp16" | "bf16"
}
```

#### Outputs
```python
{
    "panorama": torch.Tensor,         # RGB (H×W×3), range [0, 1]
    "distance": torch.Tensor,         # Depth (H×W×1), metric scale
    "normal": torch.Tensor,           # Normals (H×W×3), range [-1, 1]
    "albedo": torch.Tensor,           # Base color (H×W×3), range [0, 1]
    "roughness": torch.Tensor,        # Roughness (H×W×1), range [0, 1]
    "metallic": torch.Tensor,         # Metallic (H×W×1), range [0, 1]
    "semantic": torch.Tensor,         # Segmentation (H×W×C), one-hot
}
```

### 2.3 Performance Requirements

| Metric | Target | Notes |
|--------|--------|-------|
| Generation Time | < 30s | 1024×2048 on RTX 4090 |
| Perception Time | < 10s | All modalities |
| VRAM Usage | < 16GB | With model offloading |
| Minimum VRAM | 8GB | Reduced precision + offloading |
| Batch Size | 1-4 | Depending on VRAM |

---

## 3. Node Design

### 3.1 Node Class: OmniXModelLoader

**Purpose:** Load and cache Flux.1-dev base model and OmniX adapters

**Inputs:**
- `model_name`: str ["omnix-base", "omnix-large"]
- `precision`: str ["fp32", "fp16", "bf16"]
- `offload_mode`: str ["none", "sequential", "model"]

**Outputs:**
- `omnix_model`: Custom model object

**Implementation Notes:**
```python
class OmniXModelLoader:
    CATEGORY = "OmniX"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["omnix-base"], {"default": "omnix-base"}),
                "precision": (["fp32", "fp16", "bf16"], {"default": "fp16"}),
                "offload_mode": (["none", "sequential", "model"], {"default": "model"}),
            }
        }
    
    RETURN_TYPES = ("OMNIX_MODEL",)
    FUNCTION = "load_model"
    
    def load_model(self, model_name, precision, offload_mode):
        # Load Flux.1-dev base
        # Load OmniX adapters
        # Configure offloading
        # Return wrapped model
        pass
```

### 3.2 Node Class: OmniXPanoramaGenerator

**Purpose:** Generate 360° panorama from text/image

**Inputs:**
- `omnix_model`: OMNIX_MODEL
- `prompt`: STRING (multiline)
- `negative_prompt`: STRING (multiline, optional)
- `conditioning_image`: IMAGE (optional)
- `seed`: INT
- `steps`: INT (default: 28)
- `guidance_scale`: FLOAT (default: 5.0)
- `width`: INT [1024, 2048, 4096]
- `height`: INT [512, 1024, 2048]

**Outputs:**
- `panorama`: IMAGE (equirectangular)

**Implementation Notes:**
```python
class OmniXPanoramaGenerator:
    CATEGORY = "OmniX"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "omnix_model": ("OMNIX_MODEL",),
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "Photorealistic interior scene"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0}),
                "width": ([1024, 2048, 4096], {"default": 2048}),
                "height": ([512, 1024, 2048], {"default": 1024}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True}),
                "conditioning_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    
    def generate(self, omnix_model, prompt, seed, steps, 
                 guidance_scale, width, height, 
                 negative_prompt="", conditioning_image=None):
        # Set seed
        # Encode prompt
        # Optional: encode conditioning image
        # Run flow matching diffusion
        # Denoise and decode
        # Convert to ComfyUI IMAGE format
        pass
```

### 3.3 Node Class: OmniXPanoramaPerception

**Purpose:** Extract geometric and material properties from panorama

**Inputs:**
- `omnix_model`: OMNIX_MODEL
- `panorama`: IMAGE
- `extract_distance`: BOOLEAN
- `extract_normal`: BOOLEAN
- `extract_albedo`: BOOLEAN
- `extract_roughness`: BOOLEAN
- `extract_metallic`: BOOLEAN
- `extract_semantic`: BOOLEAN

**Outputs:**
- `distance`: IMAGE (optional)
- `normal`: IMAGE (optional)
- `albedo`: IMAGE (optional)
- `roughness`: IMAGE (optional)
- `metallic`: IMAGE (optional)
- `semantic`: IMAGE (optional)

**Implementation Notes:**
```python
class OmniXPanoramaPerception:
    CATEGORY = "OmniX"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "omnix_model": ("OMNIX_MODEL",),
                "panorama": ("IMAGE",),
                "extract_distance": ("BOOLEAN", {"default": True}),
                "extract_normal": ("BOOLEAN", {"default": True}),
                "extract_albedo": ("BOOLEAN", {"default": True}),
                "extract_roughness": ("BOOLEAN", {"default": True}),
                "extract_metallic": ("BOOLEAN", {"default": True}),
                "extract_semantic": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("distance", "normal", "albedo", "roughness", "metallic", "semantic")
    FUNCTION = "perceive"
    
    def perceive(self, omnix_model, panorama, 
                 extract_distance, extract_normal, extract_albedo,
                 extract_roughness, extract_metallic, extract_semantic):
        # Convert panorama to model format
        # Run perception adapters
        # Post-process outputs
        # Convert to ComfyUI IMAGE format
        pass
```

### 3.4 Node Class: OmniXAllInOne

**Purpose:** Combined generation + perception in single pass

**Inputs:** Combination of Generator + Perception inputs

**Outputs:** All outputs (panorama + properties)

### 3.5 Utility Nodes

#### PanoramaViewer360
- Interactive 360° preview using Three.js
- Integration with existing ComfyUI preview system

#### PanoramaToEquirectangular
- Convert between panorama formats
- Support cubemap, equirectangular, fisheye

#### PBRMaterialPacker
- Combine albedo, roughness, metallic into material map
- Export formats: glTF, USD, Blender-compatible

---

## 4. Implementation Strategy

### 4.1 Phase 1: Core Infrastructure (Week 1-2)

**Tasks:**
1. Set up project structure
   ```
   ComfyUI/custom_nodes/ComfyUI-OmniX/
   ├── __init__.py
   ├── nodes.py
   ├── omnix/
   │   ├── __init__.py
   │   ├── model_loader.py
   │   ├── generator.py
   │   ├── perceiver.py
   │   └── utils.py
   ├── models/
   │   └── .gitkeep
   ├── requirements.txt
   └── README.md
   ```

2. Implement model loader
   - Flux.1-dev integration
   - Adapter loading system
   - Memory management

3. Create basic node scaffolding
   - INPUT_TYPES definitions
   - RETURN_TYPES specifications
   - Category organization

**Deliverables:**
- Loadable custom node package
- Model loading functional
- Basic node structure in ComfyUI UI

### 4.2 Phase 2: Panorama Generation (Week 3-4)

**Tasks:**
1. Implement text-to-panorama pipeline
   - Text encoding (T5, CLIP)
   - Flow matching diffusion loop
   - Denoising and decoding

2. Add image conditioning support
   - Image encoder integration
   - Conditioning injection

3. Optimize inference
   - Mixed precision support
   - Memory-efficient attention
   - Model offloading

**Deliverables:**
- Working OmniXPanoramaGenerator node
- Example workflows
- Performance benchmarks

### 4.3 Phase 3: Panorama Perception (Week 5-6)

**Tasks:**
1. Implement adapter inference
   - Distance prediction
   - Normal estimation
   - Albedo extraction
   - PBR material prediction

2. Output processing
   - Normalization
   - Format conversion
   - Visualization

3. Multi-output handling
   - Selective execution
   - Efficient batch processing

**Deliverables:**
- Working OmniXPanoramaPerception node
- All perception modalities functional
- Output validation tests

### 4.4 Phase 4: Integration & Polish (Week 7-8)

**Tasks:**
1. Implement utility nodes
   - 360° viewer
   - Format converters
   - Material packers

2. Create example workflows
   - Text to panorama
   - Image to panorama with properties
   - Material extraction pipeline

3. Documentation
   - Installation guide
   - Node documentation
   - Tutorial workflows

4. Testing & optimization
   - Unit tests
   - Integration tests
   - Performance profiling

**Deliverables:**
- Complete node suite
- Documentation
- Example workflows
- Release candidate

---

## 5. Data Flow

### 5.1 Text-to-Panorama Flow

```
User Input (Text Prompt)
    ↓
Text Encoder (T5/CLIP)
    ↓
Text Embeddings
    ↓
Flow Matching Diffusion
    ├─ Initial Noise
    ├─ Timestep Conditioning
    └─ RGB Adapter
    ↓
Denoising Loop (28 steps)
    ↓
Latent Panorama
    ↓
VAE Decoder
    ↓
RGB Panorama (H×W×3)
    ↓
ComfyUI IMAGE Output
```

### 5.2 Panorama Perception Flow

```
Input Panorama (H×W×3)
    ↓
Image Encoder
    ↓
Feature Maps
    ↓
┌──────────────┬──────────────┬──────────────┐
│   Distance   │   Normal     │   Albedo     │
│   Adapter    │   Adapter    │   Adapter    │
└──────┬───────┴──────┬───────┴──────┬───────┘
       │              │              │
    Distance       Normal         Albedo
      Map           Map            Map
       │              │              │
       └──────────────┴──────────────┘
                     ↓
            Post-Processing
                     ↓
        ComfyUI IMAGE Outputs
```

---

## 6. Model Weight Management

### 6.1 Directory Structure

```
ComfyUI/models/
├── diffusion_models/
│   └── flux-1-dev/
│       ├── model_index.json
│       ├── vae/
│       ├── text_encoder/
│       └── unet/
├── omnix/
│   ├── omnix-base/
│   │   ├── config.json
│   │   ├── rgb_adapter.safetensors
│   │   ├── distance_adapter.safetensors
│   │   ├── normal_adapter.safetensors
│   │   ├── albedo_adapter.safetensors
│   │   ├── roughness_adapter.safetensors
│   │   └── metallic_adapter.safetensors
│   └── omnix-large/
└── vae/
```

### 6.2 Weight Loading Strategy

**Priority Order:**
1. Check local cache: `ComfyUI/models/omnix/`
2. Download from HuggingFace Hub (when available)
3. Fallback to user-provided path

**Caching:**
- Use `safetensors` format for fast loading
- Implement lazy loading for adapters
- Cache loaded models in memory (with LRU eviction)

### 6.3 Model Size Estimates

| Component | Size | Notes |
|-----------|------|-------|
| Flux.1-dev base | ~23GB | Shared with other Flux workflows |
| RGB Adapter | ~2GB | Panorama generation |
| Distance Adapter | ~1.5GB | Depth prediction |
| Normal Adapter | ~1.5GB | Normal mapping |
| Albedo Adapter | ~1.5GB | Texture extraction |
| Roughness Adapter | ~1GB | Material property |
| Metallic Adapter | ~1GB | Material property |
| **Total (all)** | **~32GB** | Disk space required |

---

## 7. Memory Optimization

### 7.1 VRAM Management Strategies

**Level 1: No Offloading (16GB+ VRAM)**
```python
- All models in VRAM
- Fastest inference
- Target: < 15s generation
```

**Level 2: Model Offloading (12GB VRAM)**
```python
- Base model in VRAM during use
- Offload to RAM between calls
- Target: < 25s generation
```

**Level 3: Sequential CPU Offload (8GB VRAM)**
```python
- Load model components sequentially
- Aggressive RAM offloading
- Target: < 45s generation
```

### 7.2 Precision Strategies

| Precision | VRAM Savings | Quality Loss | Recommended |
|-----------|--------------|--------------|-------------|
| FP32 | 0% (baseline) | None | Development |
| FP16 | ~50% | Negligible | Production |
| BF16 | ~50% | Minimal | High-end GPUs |
| INT8 | ~75% | Moderate | Low VRAM only |

### 7.3 Batch Processing

```python
# Memory-aware batching
def adaptive_batch_size(vram_available: int, resolution: tuple) -> int:
    """Calculate optimal batch size based on VRAM and resolution"""
    base_memory = estimate_model_memory()
    image_memory = estimate_image_memory(resolution)
    safety_margin = 0.2  # 20% headroom
    
    available = vram_available * (1 - safety_margin) - base_memory
    batch_size = max(1, available // image_memory)
    return batch_size
```

---

## 8. Error Handling & Edge Cases

### 8.1 Common Error Scenarios

**Out of Memory (OOM)**
```python
try:
    output = generate_panorama(...)
except torch.cuda.OutOfMemoryError:
    # Clear cache
    torch.cuda.empty_cache()
    # Enable model offloading
    enable_cpu_offload()
    # Retry with reduced batch size
    output = generate_panorama(..., batch_size=1)
```

**Model Loading Failures**
```python
try:
    model = load_omnix_model(path)
except FileNotFoundError:
    # Attempt download from HuggingFace
    model = download_and_load(model_id)
except CorruptedWeightsError:
    # Clear cache and re-download
    clear_model_cache()
    model = download_and_load(model_id, force=True)
```

**Invalid Input Dimensions**
```python
def validate_panorama(image: torch.Tensor) -> None:
    """Validate panorama has correct aspect ratio"""
    h, w = image.shape[-2:]
    aspect_ratio = w / h
    
    if not (1.8 < aspect_ratio < 2.2):  # Should be ~2:1
        raise ValueError(
            f"Invalid panorama aspect ratio {aspect_ratio:.2f}. "
            f"Expected ~2:1 (equirectangular format)"
        )
```

### 8.2 Graceful Degradation

**Progressive Feature Disable:**
1. Disable semantic segmentation (least critical)
2. Reduce resolution by 50%
3. Switch to FP16 if using FP32
4. Enable model offloading
5. Reduce batch size to 1
6. Fall back to CPU inference (last resort)

---

## 9. Testing Strategy

### 9.1 Unit Tests

```python
# tests/test_model_loader.py
def test_model_loader():
    loader = OmniXModelLoader()
    model = loader.load_model("omnix-base", "fp16", "none")
    assert model is not None
    assert model.device.type in ["cuda", "cpu"]

# tests/test_generator.py
def test_panorama_generation():
    generator = OmniXPanoramaGenerator()
    panorama = generator.generate(
        model=mock_model,
        prompt="Test scene",
        seed=42,
        steps=1,  # Fast test
        guidance_scale=5.0,
        width=512,
        height=256
    )
    assert panorama.shape == (1, 256, 512, 3)
    assert panorama.min() >= 0.0
    assert panorama.max() <= 1.0
```

### 9.2 Integration Tests

```python
# tests/test_integration.py
def test_end_to_end_workflow():
    # Load model
    loader = OmniXModelLoader()
    model = loader.load_model("omnix-base", "fp16", "none")
    
    # Generate panorama
    generator = OmniXPanoramaGenerator()
    panorama = generator.generate(
        model=model,
        prompt="Modern living room",
        seed=42,
        steps=28,
        guidance_scale=5.0,
        width=2048,
        height=1024
    )
    
    # Extract properties
    perceiver = OmniXPanoramaPerception()
    outputs = perceiver.perceive(
        model=model,
        panorama=panorama,
        extract_distance=True,
        extract_normal=True,
        extract_albedo=True,
        extract_roughness=True,
        extract_metallic=True,
        extract_semantic=False
    )
    
    # Validate outputs
    assert outputs["distance"] is not None
    assert outputs["normal"] is not None
    assert outputs["albedo"] is not None
```

### 9.3 Performance Benchmarks

```python
# benchmarks/benchmark_generation.py
import time

def benchmark_generation(resolution, precision, offload):
    times = []
    for i in range(10):
        start = time.time()
        generate_panorama(
            prompt="Test",
            width=resolution[0],
            height=resolution[1],
            precision=precision,
            offload_mode=offload
        )
        torch.cuda.synchronize()
        times.append(time.time() - start)
    
    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times)
    }
```

---

## 10. Deployment & Distribution

### 10.1 Installation Methods

**Method 1: ComfyUI Manager (Recommended)**
```bash
# In ComfyUI Manager UI:
# 1. Search "OmniX"
# 2. Click "Install"
# 3. Restart ComfyUI
```

**Method 2: Git Clone**
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/[USERNAME]/ComfyUI-OmniX.git
cd ComfyUI-OmniX
pip install -r requirements.txt
```

**Method 3: Manual Download**
```bash
# Download ZIP from GitHub
# Extract to ComfyUI/custom_nodes/ComfyUI-OmniX
# Install dependencies
pip install -r requirements.txt
```

### 10.2 Requirements File

```
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
diffusers>=0.21.0
transformers>=4.35.0
safetensors>=0.4.0
accelerate>=0.24.0
omegaconf>=2.3.0
einops>=0.7.0
pillow>=10.0.0
numpy>=1.24.0
scipy>=1.11.0
```

### 10.3 Release Checklist

**Pre-Release:**
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Example workflows tested
- [ ] Performance benchmarks documented
- [ ] Known issues documented

**Release Assets:**
- [ ] Source code (GitHub)
- [ ] Installation guide
- [ ] User documentation
- [ ] API reference
- [ ] Tutorial videos (optional)
- [ ] Pre-trained weights links

**Post-Release:**
- [ ] Monitor issue tracker
- [ ] Respond to user feedback
- [ ] Plan bug fix releases
- [ ] Community support

---

## 11. Future Enhancements

### 11.1 Phase 2 Features (Post-Launch)

**Panorama Completion (Inpainting)**
- Mask-based editing
- Region-specific regeneration
- Seamless blending

**3D Scene Generation**
- Mesh extraction from depth
- UV mapping
- Blender export integration
- Unity/Unreal Engine compatibility

**Advanced Material Editing**
- Material property adjustment
- PBR material library
- Style transfer for materials

### 11.2 Performance Optimizations

**Model Quantization**
- INT8 quantization for adapters
- Dynamic quantization during inference
- Quality vs. speed tradeoffs

**Optimized Kernels**
- Custom CUDA kernels for adapters
- Flash Attention integration
- Memory-efficient transformers

**Distributed Inference**
- Multi-GPU support
- Model parallelism
- Pipeline parallelism

### 11.3 Extended Capabilities

**Multi-View Consistency**
- Generate multiple panoramas
- Ensure geometric consistency
- Camera path planning

**Temporal Consistency**
- Video panorama generation
- Frame interpolation
- Smooth transitions

**Interactive Editing**
- Real-time preview
- Brush-based editing
- Object placement

---

## 12. Risk Assessment

### 12.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Model weights not released | High | Medium | Contact authors, fallback to training |
| VRAM requirements too high | High | Low | Implement aggressive offloading |
| Slow inference speed | Medium | Medium | Optimize, document hardware requirements |
| Compatibility issues | Medium | Low | Extensive testing, version pinning |
| Memory leaks | Low | Low | Proper resource cleanup, testing |

### 12.2 Project Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| OmniX API changes | Medium | Low | Version lock, adapter layer |
| ComfyUI breaking changes | Medium | Low | Monitor updates, maintain compatibility |
| Limited adoption | Low | Medium | Good documentation, examples |
| Competition from official implementation | Low | High | Focus on ComfyUI-specific features |

---

## 13. Success Metrics

### 13.1 Technical Metrics

- **Performance:** < 30s generation time @ 2K resolution on RTX 4090
- **Quality:** Visual parity with official OmniX implementation
- **Memory:** < 16GB VRAM with offloading enabled
- **Reliability:** < 1% error rate in production

### 13.2 User Adoption Metrics

- GitHub stars (target: 500+ in 3 months)
- ComfyUI Manager installs (target: 1000+ in 3 months)
- Community workflows created (target: 50+ in 3 months)
- Issue resolution time (target: < 48 hours)

### 13.3 Quality Metrics

- Test coverage > 80%
- Documentation completeness > 90%
- User satisfaction score > 4.5/5
- Zero critical bugs in production

---

## 14. Timeline & Milestones

```
Week 1-2:  Core Infrastructure
├─ Project setup
├─ Model loader implementation
└─ Basic node scaffolding

Week 3-4:  Panorama Generation
├─ Text-to-panorama pipeline
├─ Image conditioning
└─ Inference optimization

Week 5-6:  Panorama Perception
├─ Adapter inference
├─ Multi-modal outputs
└─ Output processing

Week 7-8:  Integration & Polish
├─ Utility nodes
├─ Documentation
├─ Example workflows
└─ Testing & optimization

Week 9:    Beta Release
├─ Community testing
├─ Bug fixes
└─ Performance tuning

Week 10:   Production Release
├─ Final testing
├─ Release documentation
└─ Public launch
```

---

## 15. Appendices

### Appendix A: Glossary

- **Equirectangular:** 2:1 aspect ratio panorama format
- **PBR:** Physically Based Rendering material properties
- **Flow Matching:** Alternative to diffusion for generative models
- **DiT:** Diffusion Transformer architecture
- **Adapter:** Lightweight module for task-specific fine-tuning

### Appendix B: References

1. OmniX Paper: https://arxiv.org/abs/2510.26800
2. OmniX GitHub: https://github.com/HKU-MMLab/OmniX
3. Flux.1-dev: https://github.com/black-forest-labs/flux
4. ComfyUI: https://github.com/comfyanonymous/ComfyUI
5. Diffusers: https://github.com/huggingface/diffusers

### Appendix C: Contact Information

**Project Lead:** [Your Name]  
**Repository:** https://github.com/[USERNAME]/ComfyUI-OmniX  
**Discord:** [Discord Server Link]  
**Email:** [Contact Email]

---

**Document Version History:**
- v1.0 (2025-11-04): Initial design document
