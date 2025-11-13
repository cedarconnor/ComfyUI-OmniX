# Changelog

All notable changes to ComfyUI-OmniX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- 🐛 Removed excessive debug logging from `omnix/cross_lora.py`
- 🐛 Fixed hardcoded Windows path in `nodes_diffusers.py` - now auto-detects cross-platform paths
- 🐛 Improved torch.load patching safety with threading lock and better error handling
- 📝 Updated README placeholder URLs to actual repository (cedarconnor/ComfyUI-OmniX)

### Changed
- 🔄 Replaced all `print()` statements with proper Python `logging` module
  - `nodes_diffusers.py`: INFO level for user-facing messages
  - `omnix/adapters.py`: INFO level for operations, DEBUG for details
  - `omnix/cross_lora.py`: INFO/DEBUG/WARNING levels appropriately

### Added
- 📚 Created `omnix/ADAPTER_IMPLEMENTATIONS.md` - comprehensive adapter architecture documentation
- 📚 Documented incomplete test status in `tests/test_e2e_workflow.py` and `tests/test_model_loader.py`
- 📚 Created `.archive/` directory for session documentation with README
- 📝 Added this CHANGELOG.md

### Removed
- 🗑️ Moved session documentation to `.archive/`:
  - PHASE1_COMPLETE.md
  - PHASE1_PROGRESS_REPORT.md
  - ROUND3_FIXES.md, ROUND4_FIXES.md
  - FIXES_APPLIED.md, FINAL_LORA_FIX.md
  - WEIGHT_CONVERSION_FIX.md
  - IMPLEMENTATION_PLAN.md, REMAINING_IMPLEMENTATION.md

## [1.1.0] - 2024 - "Diffusers-Only"

### Changed
- **🚀 Major Architecture Change**: Migrated from ComfyUI-native Flux to HuggingFace Diffusers
  - Eliminated weight conversion issues
  - Perfect compatibility with OmniX LoRA adapters
  - Simplified architecture

### Added
- ✨ `FluxDiffusersLoader` node - loads Flux.1-dev via Diffusers
- ✨ `OmniXLoRALoader` node - attaches OmniX LoRA adapters to pipeline
- ✨ `OmniXPerceptionDiffusers` node - runs perception tasks with Diffusers
- 📄 `DIFFUSERS_IMPLEMENTATION.md` - architecture documentation
- 📄 `MODEL_DOWNLOAD_GUIDE.md` - model setup instructions

### Removed
- ❌ ComfyUI-native Flux nodes (deprecated in favor of Diffusers)

## [1.0.0] - 2024 - "Initial Release"

### Added
- ✨ OmniX panorama perception for ComfyUI
- 🎨 Perception modes:
  - Distance/depth maps
  - Normal maps (surface normals)
  - Albedo maps (base color extraction)
  - PBR maps (roughness and metallic)
- 🏗️ Core architecture:
  - `omnix/adapters.py` - Adapter management
  - `omnix/perceiver.py` - Perception pipeline
  - `omnix/model_loader.py` - Model lifecycle
  - `omnix/utils.py` - Utility functions
  - `omnix/error_handling.py` - Error management
- 📝 Comprehensive documentation
- 🧪 Unit tests for core functionality
- 📦 Model download utility (`download_models.py`)
- 🔧 ComfyUI workflow examples

### Dependencies
- torch >= 2.0.0
- diffusers >= 0.34.0
- transformers >= 4.37.0
- peft >= 0.10.0
- safetensors >= 0.4.0
- accelerate >= 0.24.0

---

## Version History Summary

- **v1.1.0**: Diffusers-only implementation for perfect adapter compatibility
- **v1.0.0**: Initial release with ComfyUI-native Flux (deprecated)

## Upgrade Guide

### From v1.0.0 to v1.1.0

**Breaking Changes:**
- Old ComfyUI-native Flux nodes are no longer supported
- Workflows must be updated to use Diffusers-based nodes

**Migration Steps:**
1. Update workflow to use new nodes:
   - Replace `CheckpointLoaderSimple` → `FluxDiffusersLoader`
   - Add `OmniXLoRALoader` after Flux loader
   - Replace old perception nodes → `OmniXPerceptionDiffusers`
2. No adapter weight changes needed (same files work)
3. See `/workflows/example_perception.json` for reference

**Benefits:**
- ✅ No more weight conversion errors
- ✅ Perfect OmniX adapter compatibility
- ✅ Simplified architecture
- ✅ Better error messages

---

## Contributing

When contributing, please:
1. Update this CHANGELOG under `[Unreleased]` section
2. Follow [Conventional Commits](https://www.conventionalcommits.org/)
3. Group changes by type: Added, Changed, Deprecated, Removed, Fixed, Security

## Links

- **Repository**: https://github.com/cedarconnor/ComfyUI-OmniX
- **Issues**: https://github.com/cedarconnor/ComfyUI-OmniX/issues
- **Discussions**: https://github.com/cedarconnor/ComfyUI-OmniX/discussions

---

**Legend:**
- ✨ New feature
- 🐛 Bug fix
- 🔄 Change
- ❌ Removal
- 🚀 Major change
- 📝 Documentation
- 🧪 Tests
- 🔧 Configuration
- 📚 Documentation improvement
- 🗑️ Deprecated/Archived
