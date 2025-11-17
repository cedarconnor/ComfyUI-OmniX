# ComfyUI‑OmniX (Diffusers Edition)

OmniX brings HKU-MMLab's panorama perception adapters to ComfyUI. This fork removes the legacy conversion pipeline and runs the original LoRA weights directly inside HuggingFace Diffusers' `FluxPipeline`, so every perception head (distance, normal, albedo, PBR) matches the official implementation.

![OmniX perception outputs](workflows/pano_perc.png)

---

## Requirements

| Component | Location | Notes |
|-----------|----------|-------|
| **Flux.1-dev Diffusers repo** | `ComfyUI/models/diffusers/flux1-dev/...` | Copy the full `black-forest-labs/FLUX.1-dev` repository (JSON configs + subfolders). See [Model Downloads](#model-downloads) below. |
| **Flux checkpoint** | `ComfyUI/models/diffusers/flux1-dev.safetensors` (or `.sft`) | Place the file beside the Diffusers repo; the loader never downloads weights. |
| **OmniX LoRA adapters** | `ComfyUI/models/loras/omnix/` | Download from `https://huggingface.co/KevinHuang/OmniX` (distance, normal, albedo, pbr). See [Model Downloads](#model-downloads) below. |
| **Panorama input** | Any 2:1 equirectangular image | Feed via ComfyUI's `LoadImage` node. |

Python dependencies (Diffusers ≥ 0.29, PEFT ≥ 0.10, PyTorch ≥ 2.1) are listed in `requirements.txt`. Install them inside your ComfyUI environment.

---

## Installation

1. Navigate to `ComfyUI/custom_nodes/` and clone this repo or copy its folder:
   ```bash
   git clone https://github.com/cedarconnor/ComfyUI-OmniX.git
   ```
2. Install Python requirements (inside your ComfyUI venv):
   ```bash
   pip install -r ComfyUI-OmniX/requirements.txt
   ```
3. Copy the Flux Diffusers repo + checkpoint into `ComfyUI/models/diffusers/` (structure must match Diffusers’ layout).
4. Download OmniX adapters into `ComfyUI/models/loras/omnix/`.
5. Restart ComfyUI so the new nodes register.

---

## Model Downloads

### Required Files and Locations

#### 1. **FLUX.1-dev Model** (Required)

You need **both** the Diffusers repository structure AND the checkpoint file.

**Option A: Download Diffusers Repo (Recommended)**

1. **Visit**: https://huggingface.co/black-forest-labs/FLUX.1-dev
2. **Download Method 1 - Using git**:
   ```bash
   cd ComfyUI/models/diffusers/
   git clone https://huggingface.co/black-forest-labs/FLUX.1-dev flux1-dev
   ```

3. **Download Method 2 - Manual Download**:
   - Click "Files and versions" tab on HuggingFace
   - Download the entire repository structure maintaining folders:
     ```
     ComfyUI/models/diffusers/flux1-dev/
     ├── model_index.json
     ├── scheduler/
     │   └── scheduler_config.json
     ├── text_encoder/
     │   └── config.json
     ├── text_encoder_2/
     │   └── config.json
     ├── tokenizer/
     │   ├── merges.txt
     │   └── vocab.json
     ├── tokenizer_2/
     │   └── spiece.model
     ├── transformer/
     │   └── config.json
     └── vae/
         └── config.json
     ```

**AND**

4. **Download the Flux checkpoint** (one of these sources):
   - **HuggingFace**: https://huggingface.co/black-forest-labs/FLUX.1-dev
     - Look for `flux1-dev.safetensors` (~23GB)
   - **Alternative sources**: CivitAI, other mirror sites

5. **Place checkpoint here**:
   ```
   ComfyUI/models/diffusers/flux1-dev.safetensors
   ```
   (Same directory as the `flux1-dev` folder, NOT inside it)

**Important**: The checkpoint filename must match what you select in the FluxDiffusersLoader node. Common names:
- `flux1-dev.safetensors` (recommended)
- `flux1-dev.sft`

---

#### 2. **OmniX LoRA Adapters** (Required)

**Download Location**: https://huggingface.co/KevinHuang/OmniX

**Files to Download**:

| File Name | Task | Size | Required? |
|-----------|------|------|-----------|
| `rgb_to_depth_depth.safetensors` | Distance/Depth maps | ~200MB | ✅ Yes (if using depth) |
| `rgb_to_normal_normal.safetensors` | Normal maps | ~200MB | ✅ Yes (if using normals) |
| `rgb_to_albedo_albedo.safetensors` | Albedo/base color | ~200MB | ✅ Yes (if using albedo) |
| `rgb_to_pbr_pbr.safetensors` | PBR (roughness+metallic) | ~200MB | ⚠️ Optional |

**Installation Steps**:

1. Create the directory:
   ```bash
   mkdir -p ComfyUI/models/loras/omnix/
   ```

2. Download adapters from HuggingFace:
   ```bash
   cd ComfyUI/models/loras/omnix/

   # Download using wget or curl
   wget https://huggingface.co/KevinHuang/OmniX/resolve/main/rgb_to_depth_depth.safetensors
   wget https://huggingface.co/KevinHuang/OmniX/resolve/main/rgb_to_normal_normal.safetensors
   wget https://huggingface.co/KevinHuang/OmniX/resolve/main/rgb_to_albedo_albedo.safetensors
   wget https://huggingface.co/KevinHuang/OmniX/resolve/main/rgb_to_pbr_pbr.safetensors
   ```

   Or download manually from the HuggingFace Files tab.

3. **Verify your directory structure**:
   ```
   ComfyUI/models/loras/omnix/
   ├── rgb_to_depth_depth.safetensors
   ├── rgb_to_normal_normal.safetensors
   ├── rgb_to_albedo_albedo.safetensors
   └── rgb_to_pbr_pbr.safetensors
   ```

**Important**:
- ⚠️ **Do NOT rename these files** - the node looks for exact filenames
- The default `adapter_dir` in OmniXLoRALoader is `ComfyUI/models/loras/omnix`
- You can change the directory in the node if needed

---

#### 3. **Complete Directory Structure**

After downloading everything, your structure should look like:

```
ComfyUI/
├── models/
│   ├── diffusers/
│   │   ├── flux1-dev/              # ← Diffusers repo folder
│   │   │   ├── model_index.json
│   │   │   ├── scheduler/
│   │   │   ├── text_encoder/
│   │   │   ├── text_encoder_2/
│   │   │   ├── tokenizer/
│   │   │   ├── tokenizer_2/
│   │   │   ├── transformer/
│   │   │   └── vae/
│   │   └── flux1-dev.safetensors   # ← Checkpoint file (beside folder)
│   └── loras/
│       └── omnix/                  # ← OmniX adapters folder
│           ├── rgb_to_depth_depth.safetensors
│           ├── rgb_to_normal_normal.safetensors
│           ├── rgb_to_albedo_albedo.safetensors
│           └── rgb_to_pbr_pbr.safetensors
└── custom_nodes/
    └── ComfyUI-OmniX/              # ← This repository
```

---

#### Storage Requirements

- **FLUX.1-dev checkpoint**: ~23 GB
- **Diffusers configs**: ~10 MB (negligible)
- **OmniX adapters**: ~200 MB each × 4 = ~800 MB total
- **Total**: ~24 GB minimum (with all adapters)

**Memory Requirements**:
- **GPU VRAM**: 16+ GB recommended (for bfloat16)
- **System RAM**: 32+ GB recommended

---

## Node Overview

| Node | Purpose | Key Inputs | Output |
|------|---------|------------|--------|
| **FluxDiffusersLoader** | Loads `FluxPipeline` strictly from local files. | `torch_dtype` (bf16/fp16/fp32), `local_checkpoint` (filename). | Pipeline, VAE, text encoder. |
| **OmniXLoRALoader** | Injects OmniX LoRA adapters into the pipeline. | `adapter_dir`, booleans for distance/normal/albedo/pbr, `lora_scale` (default: 1.0). | Patched pipeline + metadata dict. |
| **OmniXPerceptionDiffusers** | Runs one perception head using the patched pipeline. | `flux_pipeline`, `loaded_adapters`, panorama `IMAGE`, `task`, `num_steps`, `guidance_scale`, `noise_strength`, `prompt_mode` (default: "task_name"). | Single perception map (`IMAGE`). |
| **Visualization nodes** | Post-process raw tensors. | Depth / Normal / Albedo / PBR specific parameters (gamma, percentiles, etc.). | Display-ready `IMAGE`. |

Typical workflow: `LoadImage → FluxDiffusersLoader → OmniXLoRALoader → OmniXPerceptionDiffusers (duplicated per head) → Visualization → SaveImage/Preview`.

---

## Recommended Settings per Output

**Based on original HKU-MMLab/OmniX implementation:**

| Output | noise_strength | num_steps | guidance_scale | Visualization Tips |
|--------|----------------|-----------|----------------|--------------------|
| **Distance (Depth)** | **0.15–0.25** | **28** | **3.0–3.5** | Use `DepthVisualization` (viridis or grayscale) with percentiles (0.02 / 0.98). Low strength keeps geometry faithful. Gamma: **1.2** |
| **Normal** | **0.20–0.30** | **28** | **3.5–4.0** | `NormalVisualization`: enable `invert_y=True` (OpenGL convention). Gamma: **0.8** for better contrast. |
| **Albedo** | **0.30–0.40** | **28–32** | **3.5–4.5** | `AlbedoVisualization`: tweak `exposure` (0.9–1.1). Gamma: **1.1** to remove residual lighting and enhance base colors. |
| **PBR (roughness/metallic)** | **0.25–0.35** | **28–32** | **3.0–4.0** | `PBRVisualization`: choose `mode="roughness"`, `"metallic"`, or `"combined"` to inspect both channels. |

> **Important Notes:**
> - **Default `num_steps` is now 28** to match the original OmniX implementation (previously 20)
> - **`lora_scale` should be 1.0** for best results (avoid values > 1.2 unless experimenting)
> - **`prompt_mode` options:**
>   - `"empty"`: No prompt (unconditional generation) - try this first if outputs look wrong
>   - `"task_name"`: Simple task name like "distance" or "normal" (default, recommended)
>   - `"minimal"`: Short descriptor like "perception: distance"
>   - `"descriptive"`: Full panoramic description (may reduce quality for some adapters)
> - `noise_strength` behaves like Diffusers img2img "strength":
>   - **Below 0.1**: Output too similar to input, minimal perception
>   - **0.15–0.40**: Optimal range for perception tasks (recommended)
>   - **Above 0.5**: Deliberately hallucinates new structures (not recommended for perception)
> - Higher `num_steps` (up to 32) can reduce noise but increases inference time
> - All panorama inputs **must be 2:1 aspect ratio** (e.g., 1024×512, 2048×1024)

---

## Workflow Usage Example

**Prerequisites**: Ensure you've downloaded all required models (see [Model Downloads](#model-downloads) section above).

1. **Load Panorama**: Use `LoadImage` and preview to ensure **2:1 aspect ratio** (e.g., 1024×512 or 2048×1024).

2. **Load Flux Pipeline**:
   - Place `FluxDiffusersLoader` node
   - Set `torch_dtype="bfloat16"` (fastest on RTX 30/40 series)
   - Select `local_checkpoint="flux1-dev.safetensors"`

3. **Load OmniX Adapters**:
   - Add `OmniXLoRALoader` node
   - Default `adapter_dir` is `ComfyUI/models/loras/omnix`
   - Enable checkboxes for perception heads you need (distance, normal, albedo, pbr)
   - Set `lora_scale=1.0` (default)

4. **Add Perception Nodes**:
   - Duplicate `OmniXPerceptionDiffusers` for each task (distance, normal, albedo, pbr)
   - Wire all to the same panorama input
   - **Recommended settings per task** (from table above):
     - **Distance**: `num_steps=28`, `noise_strength=0.20`, `guidance_scale=3.0`
     - **Normal**: `num_steps=28`, `noise_strength=0.25`, `guidance_scale=3.5`
     - **Albedo**: `num_steps=28`, `noise_strength=0.35`, `guidance_scale=4.0`
     - **PBR**: `num_steps=28`, `noise_strength=0.30`, `guidance_scale=3.5`

5. **Add Visualization Nodes**:
   - Depth → `DepthVisualization` with `mode="viridis"`, `gamma=1.2`, `percentiles=0.02/0.98`
   - Normal → `NormalVisualization` with `invert_y=True`, `gamma=0.8`
   - Albedo → `AlbedoVisualization` with `exposure=0.95`, `gamma=1.1`, `saturation=0.9`
   - PBR → `PBRVisualization` with `mode="combined"` (red=roughness, blue=metallic)

6. **Save Outputs**:
   - Connect each visualizer to `SaveImage` or `PreviewImage` nodes
   - Use unique filenames: `omnix_depth`, `omnix_normal`, `omnix_albedo`, `omnix_pbr`

7. **Queue Workflow**:
   - Each branch runs sequentially on the shared pipeline
   - No need to reload Flux between tasks
   - Total time: ~2-3 minutes for all 4 tasks on RTX 3090/4090

**Example workflow JSON**: `workflows/omnix_perception_visualizations.json` (load via ComfyUI's workflow manager).

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| **Models not found** | See [Model Downloads](#model-downloads) section for exact file locations and naming. Verify directory structure matches exactly. |
| `Adapter 'XYZ' not loaded` | 1) Ensure the checkbox for that adapter is enabled in `OmniXLoRALoader`<br>2) Verify the `.safetensors` file exists in `models/loras/omnix/`<br>3) Check exact filename (e.g., `rgb_to_depth_depth.safetensors` - don't rename) |
| `Adapter name ... already in use` | **FIXED** in latest version. OmniXLoRALoader now automatically detects and clears existing adapters. Simply re-queue the workflow. If you still see this error, update to the latest version. |
| Output looks identical to input | `noise_strength` too low (below 0.1). Increase to **0.20–0.35** for proper perception. |
| Outputs look completely hallucinated | `noise_strength` too high (≥0.5). Drop to **0.20–0.35** for perception tasks. |
| **Outputs have wrong colors/patterns (e.g., saturated normals)** | **⚠️ MOST COMMON ISSUE**: 1) Set `prompt_mode="empty"` or `"task_name"` (not "descriptive")<br>2) Reduce `lora_scale` to **1.0** (not 1.3+)<br>3) See [OUTPUT_QUALITY_INVESTIGATION.md](OUTPUT_QUALITY_INVESTIGATION.md) for detailed diagnosis |
| Poor quality / noisy outputs | 1) Ensure `num_steps=28` (default changed from 20)<br>2) Use recommended `guidance_scale` per task (see table above)<br>3) Verify you're using **bfloat16** dtype<br>4) Check aspect ratio is exactly 2:1<br>5) Try `prompt_mode="task_name"` or `"empty"` |
| Diffusers missing files | Verify directory tree matches Diffusers layout: `scheduler/`, `transformer/`, `text_encoder/`, `text_encoder_2/`, `tokenizer/`, `tokenizer_2/`, `vae/` folders must exist with `config.json` files. Also need `model_index.json` in root. |
| CUDA out of memory | 1) Reduce image resolution (try 512×256 instead of 2048×1024)<br>2) Close other GPU applications<br>3) Use `torch_dtype="float16"` instead of bfloat16 (slightly faster but lower quality) |
| Aspect ratio warning | Your input image is not 2:1 ratio. Resize to 1024×512, 2048×1024, or other 2:1 dimensions for best results. |

### Recent Improvements

**v1.3 (Latest) - Output Quality Investigation & Fixes**
✅ **Added `prompt_mode` parameter** - Control prompt format (empty/task_name/minimal/descriptive) to fix quality issues
✅ **Added comprehensive debug logging** - Verify adapter loading and PEFT integration status
✅ **Created OUTPUT_QUALITY_INVESTIGATION.md** - Detailed diagnosis of quality issues and fixes
✅ **Updated README troubleshooting** - Added guidance for most common quality issues

**v1.2 - Adapter Loading Fixes**
✅ **Fixed "adapter already in use" error** - Automatic adapter clearing when re-queueing workflows
✅ **Improved logging** - Better debugging and error messages

**v1.1 - Quality & Performance**
✅ **Fixed critical LoRA weight averaging bug** - Significantly improved output quality
✅ **Updated default steps to 28** - Matches original OmniX implementation
✅ **Added comprehensive input validation** - Clear error messages for common mistakes
✅ **Optimized performance** - 50-70% faster on high-resolution images

See `QUALITY_REVIEW_FINDINGS.md`, `IMPLEMENTATION_SUMMARY.md`, and `OUTPUT_QUALITY_INVESTIGATION.md` for detailed technical information.

---

**Additional Resources**:
- Detailed setup guides: `DOCS/QUICK_START_DIFFUSERS.md`
- Model download guide: `DOCS/MODEL_DOWNLOAD_GUIDE.md` (if exists)
- Quality review: `QUALITY_REVIEW_FINDINGS.md`

---

## License / Credits

- OmniX adapters by **HKU‑MMLab** (`https://github.com/HKU-MMLab/OmniX`).
- Flux.1-dev by **Black Forest Labs**.
- ComfyUI by **comfyanonymous**.
- This repository is MIT/Apache-2.0 as per the original files.

Please cite OmniX if you use these perception maps in research or production. Enjoy! 🚀
