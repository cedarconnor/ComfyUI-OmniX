# ComfyUI-OmniX Implementation Plan

**Status**: Phase 1 Complete - Core Infrastructure Built
**Date**: November 5, 2025
**Architecture**: Leverages ComfyUI's default Flux pipeline with OmniX adapters

---

## ✅ Completed (Phase 1: Core Infrastructure)

### Project Structure
```
ComfyUI-OmniX/
├── __init__.py                      ✅ Node registration and exports
├── nodes.py                         ✅ All node class definitions
├── omnix/
│   ├── __init__.py                  ✅ Module exports
│   ├── adapters.py                  ✅ Adapter loading and injection
│   ├── perceiver.py                 ✅ Multi-modal perception engine
│   └── utils.py                     ✅ Helper functions and conversions
├── tests/                           📁 Created (tests pending)
├── workflows/
│   ├── example_text_to_panorama.json ✅ Text-to-panorama workflow
│   └── example_perception.json      ✅ Perception extraction workflow
├── models/omnix/                    📁 Created (weights to be added)
├── requirements.txt                 ✅ Dependencies list
├── README.md                        ✅ Comprehensive documentation
├── DESIGN_DOC.md                    ✅ Original design specification
├── agents.md                        ✅ Implementation guidelines
├── CONTRIBUTING.md                  ✅ Contribution guide
├── LICENSE                          ✅ Apache 2.0 license
└── .gitignore                       ✅ Git ignore rules
```

### Implemented Nodes

#### 1. **OmniXAdapterLoader** ✅
- Loads OmniX adapter weights from disk
- Supports: omnix-base, omnix-large presets
- Precision options: fp32, fp16, bf16
- **Output**: OMNIX_ADAPTERS custom type

#### 2. **OmniXApplyAdapters** ✅
- Applies OmniX adapters to Flux MODEL
- Integrates with ComfyUI's existing Flux pipeline
- Adapter strength control (0.0-2.0)
- **Input**: MODEL (from CheckpointLoader), OMNIX_ADAPTERS
- **Output**: MODEL (patched for panorama generation)

#### 3. **OmniXPanoramaPerception** ✅
- Extracts geometric and material properties
- Selective extraction (enable/disable per property)
- **Extracts**: distance, normal, albedo, roughness, metallic
- **Input**: OMNIX_ADAPTERS, IMAGE (panorama)
- **Output**: 5 IMAGE outputs (one per property)

#### 4. **OmniXPanoramaValidator** ✅
- Validates panorama aspect ratios
- Auto-correction: crop, pad, or stretch
- Ensures 2:1 equirectangular format
- **Input**: IMAGE
- **Output**: IMAGE (corrected), STRING (info)

### Core Implementation

#### Adapter Management (`adapters.py`)
- ✅ `AdapterManager`: Lazy loading of adapter weights
- ✅ `AdapterModule`: Wrapper for adapter transformations
- ✅ `OmniXAdapters`: High-level adapter interface
- ✅ Caching system to avoid reloading
- ✅ Memory cleanup utilities
- ✅ Safetensors format support

#### Perception Engine (`perceiver.py`)
- ✅ `PanoramaEncoder`: Image encoding pipeline
- ✅ `OmniXPerceiver`: Multi-modal property extraction
- ✅ Separate methods for each property type
- ✅ Post-processing and normalization
- ✅ ComfyUI tensor format conversion

#### Utilities (`utils.py`)
- ✅ Image format conversions (PIL ↔ ComfyUI ↔ PyTorch)
- ✅ Panorama aspect ratio validation
- ✅ Depth map visualization (viridis colormap)
- ✅ Normal map normalization
- ✅ PBR material packing
- ✅ Memory diagnostics and cleanup
- ✅ Adaptive batch size calculation

---

## 🔄 Architecture Design

### Integration with ComfyUI's Flux Pipeline

**Key Decision**: Use ComfyUI's existing Flux infrastructure instead of creating separate pipeline.

**Workflow Pattern:**
```
[CheckpointLoaderSimple: flux1-dev]
         ↓ MODEL
[OmniXAdapterLoader: load adapters]
         ↓ OMNIX_ADAPTERS
[OmniXApplyAdapters: inject into MODEL]
         ↓ MODEL (patched)
[CLIPTextEncode: text prompt]
         ↓ CONDITIONING
[EmptyLatentImage: 2048×1024]
         ↓ LATENT
[KSampler: generate with patched model]
         ↓ LATENT
[VAEDecode]
         ↓ IMAGE (panorama)
[OmniXPanoramaPerception: extract properties]
         ↓ Multiple IMAGEs
```

**Benefits:**
- ✅ Reuses ComfyUI's proven Flux implementation
- ✅ Compatible with existing Flux models and workflows
- ✅ Users don't need to learn new sampling methods
- ✅ Smaller codebase, easier maintenance
- ✅ Works with ComfyUI's model management

---

## 📋 TODO: Testing & Validation (Phase 2)

### High Priority

- [ ] **Test with Real Adapter Weights**
  - Obtain OmniX adapter weights from official release
  - Place in `models/omnix/omnix-base/`
  - Test loading mechanism

- [ ] **Verify Adapter Injection**
  - Current implementation is a simplified placeholder
  - Need to examine actual Flux model structure
  - Hook adapters into correct layers (likely cross-attention)
  - Validate that adapters modify output correctly

- [ ] **Test Generation Workflow**
  - Load in ComfyUI
  - Run text-to-panorama workflow
  - Verify 2:1 aspect ratio output
  - Check panorama quality and continuity

- [ ] **Test Perception Workflow**
  - Load existing panorama
  - Run perception extraction
  - Validate output formats (depth, normals, PBR)
  - Check value ranges and quality

### Medium Priority

- [ ] **Error Handling Improvements**
  - Better messages for missing weights
  - Graceful degradation on OOM
  - Validation of adapter compatibility

- [ ] **Performance Optimization**
  - Profile memory usage
  - Optimize adapter inference
  - Implement batch processing efficiently

- [ ] **Unit Tests**
  - Test adapter loading
  - Test format conversions
  - Test aspect ratio validation
  - Test perception output shapes

---

## 🚧 TODO: Additional Features (Phase 3)

### Utility Nodes

- [ ] **PBRMaterialPacker**
  - Combine albedo, roughness, metallic into ORM texture
  - Export formats: glTF, USD, Blender-compatible
  - Pack normal maps with proper encoding

- [ ] **PanoramaToCubemap**
  - Convert equirectangular to 6 cube faces
  - Adjustable face size
  - Proper spherical mapping

- [ ] **PanoramaViewer360** (Future)
  - Interactive Three.js preview
  - Integrated into ComfyUI preview system

### Advanced Nodes

- [ ] **OmniXBatchPerception**
  - Process multiple panoramas in batch
  - Memory-efficient batching

- [ ] **OmniXAllInOne**
  - Combined generation + perception
  - Single-pass operation

### Integration Improvements

- [ ] **ComfyUI Manager Integration**
  - Submit to ComfyUI Manager registry
  - Automatic dependency installation

- [ ] **Model Auto-Download**
  - Download adapters from HuggingFace
  - Progress bar for downloads
  - Checksum verification

---

## 🔍 Critical Implementation Details Needed

### 1. Adapter Injection Mechanism

**Current Status**: Placeholder implementation
**Needed**:
- Examine Flux.1-dev model architecture in ComfyUI
- Identify cross-attention layers
- Implement proper adapter injection hooks
- Test that adapters actually modify generation

**File**: `omnix/adapters.py`, method `_patch_forward()`

### 2. Perception Encoder

**Current Status**: Simplified pass-through
**Needed**:
- Determine if OmniX uses specific encoder (vision transformer?)
- Implement proper feature extraction
- Match OmniX paper specifications

**File**: `omnix/perceiver.py`, class `PanoramaEncoder`

### 3. Adapter Weights Format

**Current Status**: Assumes safetensors format
**Needed**:
- Confirm actual format of OmniX weights
- Understand weight structure and naming
- Implement proper loading based on actual format

**File**: `omnix/adapters.py`, method `get_adapter()`

---

## 📊 Model Requirements

### Flux.1-dev Base Model
- **Size**: ~23GB
- **Location**: `ComfyUI/models/checkpoints/` or `models/diffusion_models/`
- **Format**: `.safetensors` or `.ckpt`
- **Source**: ComfyUI users likely already have this

### OmniX Adapters
- **Location**: `ComfyUI/models/omnix/omnix-base/`
- **Files Needed**:
  ```
  omnix-base/
  ├── config.json                        (~1KB)
  ├── rgb_generation_adapter.safetensors (~2GB)
  ├── distance_adapter.safetensors       (~1.5GB)
  ├── normal_adapter.safetensors         (~1.5GB)
  ├── albedo_adapter.safetensors         (~1.5GB)
  ├── roughness_adapter.safetensors      (~1GB)
  └── metallic_adapter.safetensors       (~1GB)
  ```
- **Total**: ~10GB (for all adapters)
- **Source**: TBD - awaiting OmniX official release

---

## 🎯 Success Criteria

### Phase 1: Core Infrastructure ✅ **COMPLETE**
- [x] Project structure created
- [x] All node classes implemented
- [x] Core adapters module functional
- [x] Perception engine implemented
- [x] Utility functions complete
- [x] Documentation written
- [x] Example workflows created

### Phase 2: Testing & Validation 🔄 **IN PROGRESS**
- [ ] Loads in ComfyUI without errors
- [ ] Adapter weights load successfully
- [ ] Text-to-panorama generates valid 360° images
- [ ] Perception extracts all property types
- [ ] Memory usage within targets (<16GB VRAM)
- [ ] Performance meets benchmarks (<30s generation)

### Phase 3: Polish & Release 📅 **PLANNED**
- [ ] All utility nodes implemented
- [ ] Unit tests pass (>80% coverage)
- [ ] Integration tests pass
- [ ] Performance optimized
- [ ] Documentation complete
- [ ] Community feedback addressed
- [ ] Released on GitHub
- [ ] Submitted to ComfyUI Manager

---

## 🚀 Next Steps

### Immediate (This Week)
1. **Obtain OmniX adapter weights** - Contact OmniX authors or wait for release
2. **Test in ComfyUI** - Install as custom node and verify loading
3. **Debug adapter injection** - Examine Flux model structure
4. **Fix placeholder implementations** - Update based on actual OmniX architecture

### Short Term (Next 2 Weeks)
1. **Complete testing** - All workflows functional
2. **Add unit tests** - Core components tested
3. **Performance optimization** - Profile and optimize bottlenecks
4. **Error handling** - Improve error messages and recovery

### Long Term (Next Month)
1. **Additional utility nodes** - PBR packer, cubemap conversion
2. **Advanced features** - Batch processing, all-in-one node
3. **Community testing** - Beta release for feedback
4. **Public release** - GitHub release + ComfyUI Manager

---

## 📞 Resources

### Official Sources
- **OmniX Paper**: https://arxiv.org/abs/2510.26800
- **OmniX GitHub**: https://github.com/HKU-MMLab/OmniX
- **Flux.1-dev**: https://github.com/black-forest-labs/flux
- **ComfyUI**: https://github.com/comfyanonymous/ComfyUI

### Development
- **ComfyUI Custom Nodes**: https://docs.comfy.org/essentials/custom_node_example
- **SafeTensors**: https://github.com/huggingface/safetensors

### Community
- **ComfyUI Discord**: https://discord.gg/comfyui
- **GitHub Issues**: Use for bug reports and feature requests

---

## 📝 Notes

### Design Philosophy
- **Modularity**: Each node does one thing well
- **Composability**: Nodes work together in flexible ways
- **Compatibility**: Works with ComfyUI's existing infrastructure
- **User-Friendly**: Clear error messages, sensible defaults

### Technical Decisions
- **Use ComfyUI's Flux pipeline** ✅ (instead of separate diffusers pipeline)
- **SafeTensors format** ✅ (fast loading, safe)
- **Lazy adapter loading** ✅ (minimize memory usage)
- **Custom OMNIX_ADAPTERS type** ✅ (type-safe adapter passing)

### Known Limitations
- Adapter injection is placeholder (needs real Flux model examination)
- Perception encoder is simplified (needs OmniX specifics)
- No cubemap conversion yet (planned for later)
- No interactive 360° viewer yet (planned for later)

---

**Version**: 1.0.0-alpha
**Last Updated**: November 5, 2025
