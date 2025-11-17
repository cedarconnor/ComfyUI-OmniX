# ComfyUI_OmniX

Experimental ComfyUI node pack that integrates **OmniX** panoramic generation + perception on top of **FLUX.1-dev**.

This repo is meant to be used together with:

- `design_doc_omnix_comfyui.md` – overall architecture and node design
- `agents_omnix_comfyui.md` – roles / workflow for building and maintaining the pack

---

## 1. What This Node Pack Aims To Do

- Use **FLUX.1-dev** as the base model.
- Add **OmniX LoRAs** (converted to Comfy format) for:
  - `text_to_pano` – text → RGB panorama
  - `rgb_to_depth` – pano → depth
  - `rgb_to_normal` – pano → surface normals
  - `rgb_to_albedo` – pano → albedo
  - `rgb_to_pbr` – pano → roughness / metal
  - `rgb_to_semantic` – pano → semantic segmentation
- Expose simple Comfy nodes:
  - `OmniX_PanoPerception_Depth`
  - `OmniX_PanoPerception_Normal`
  - `OmniX_PanoPerception_PBR`
  - `OmniX_PanoPerception_Semantic`

> **Note:** The Python in `omni_x_nodes.py` is intentionally conservative and uses **stub implementations** for perception. You are expected to wire in the real OmniX logic from the official repo (e.g., `run_pano_perception.py`) via `omni_x_utils.py` or similar.

---

## 2. Installation

1. Create a folder under your ComfyUI custom nodes, for example:

   ```text
   ComfyUI/
     custom_nodes/
       ComfyUI_OmniX/
         __init__.py
         omni_x_nodes.py
         omni_x_utils.py         # you add this
         design_doc_omnix_comfyui.md   # optional, docs
         agents_omnix_comfyui.md       # optional, docs
   ```

2. Copy the provided `omni_x_nodes.py` into that folder.

3. Create a minimal `__init__.py` that exposes the node classes:

   ```python
   from .omni_x_nodes import (
       OmniX_PanoPerception_Depth,
       OmniX_PanoPerception_Normal,
       OmniX_PanoPerception_PBR,
       OmniX_PanoPerception_Semantic,
   )

   NODE_CLASS_MAPPINGS = {
       "OmniX_PanoPerception_Depth": OmniX_PanoPerception_Depth,
       "OmniX_PanoPerception_Normal": OmniX_PanoPerception_Normal,
       "OmniX_PanoPerception_PBR": OmniX_PanoPerception_PBR,
       "OmniX_PanoPerception_Semantic": OmniX_PanoPerception_Semantic,
   }

   NODE_DISPLAY_NAME_MAPPINGS = {
       "OmniX_PanoPerception_Depth": "OmniX Pano Perception: Depth",
       "OmniX_PanoPerception_Normal": "OmniX Pano Perception: Normal",
       "OmniX_PanoPerception_PBR": "OmniX Pano Perception: PBR",
       "OmniX_PanoPerception_Semantic": "OmniX Pano Perception: Semantic",
   }
   ```

4. Restart ComfyUI.

If everything is wired correctly, you should see a new category **“OmniX/Perception”** (or whatever you put in `CATEGORY`) with the OmniX nodes.

---

## 3. Models and LoRAs

Place your Flux base checkpoint and OmniX LoRAs into the usual Comfy model directories, e.g.:

```text
ComfyUI/
  models/
    checkpoints/
      flux1-dev.safetensors
    loras/
      OmniX_text_to_pano_rgb_comfyui.safetensors
      OmniX_rgb_to_depth_comfyui.safetensors
      OmniX_rgb_to_normal_comfyui.safetensors
      OmniX_rgb_to_albedo_comfyui.safetensors
      OmniX_rgb_to_pbr_comfyui.safetensors
      OmniX_rgb_to_semantic_comfyui.safetensors
```

The LoRAs should already be converted from Diffusers/PEFT format into native Comfy FLUX LoRA format (key names like `lora_unet_double_blocks_*`).

---

## 4. Node Overview (High-Level)

### 4.1. OmniX_PanoPerception_Depth

- **Inputs**
  - `model` – Flux model + `rgb_to_depth` LoRA
  - `pano_latent` – pano latent from `VAEEncode`
  - `normalize_output` – whether to normalize for preview

- **Outputs**
  - `depth_img` – depth visualization as an IMAGE (float 0–1)

### 4.2. OmniX_PanoPerception_Normal

- **Inputs**
  - `model` – Flux model + `rgb_to_normal` LoRA
  - `pano_latent`
  - `normalize_output`

- **Outputs**
  - `normal_img` – RGB normal map IMAGE

### 4.3. OmniX_PanoPerception_PBR

- **Inputs**
  - `model` – Flux model + `rgb_to_pbr` LoRA
  - `pano_latent`
  - `normalize_output`

- **Outputs**
  - `albedo_img`, `roughness_img`, `metallic_img` – PBR maps as IMAGEs

### 4.4. OmniX_PanoPerception_Semantic

- **Inputs**
  - `model` – Flux model + `rgb_to_semantic` LoRA
  - `pano_latent`
  - `palette` (future option)

- **Outputs**
  - `semantic_img` – colorized semantic segmentation IMAGE

For detailed behavior and data flow, see `design_doc_omnix_comfyui.md`.

---

## 5. Quickstart Workflows

### 5.1. Text → Pano

Use standard Flux nodes plus your `text_to_pano` LoRA:

1. `CLIPTextEncode` / `CLIPTextEncode` (neg)
2. `EmptyLatentImage` with 2:1 resolution (e.g., 1024×2048)
3. `Load Diffusion Model` → Flux dev
4. `LoraLoaderModelOnly` → apply `OmniX_text_to_pano_rgb_comfyui.safetensors`
5. `KSampler` → sample the pano latent
6. `VAEDecode` → `pano_rgb`

### 5.2. Pano → Perception

1. Feed `pano_rgb` into `VAEEncode` → `pano_latent`.
2. Load Flux dev again (or reuse the same model).
3. Apply the relevant OmniX perception LoRA with `LoraLoaderModelOnly`.
4. Feed the result into the corresponding `OmniX_PanoPerception_*` node(s).
5. Connect outputs to `PreviewImage` or `ImageSave`.

### 5.3. Full Pipeline

Just chain the above:

- Text → Pano → VAEEncode → multiple `OmniX_PanoPerception_*` nodes in parallel.

---

## 6. Wiring in the Real OmniX Logic

Right now, the `omni_x_nodes.py` file contains placeholder logic so it is safe to import in Comfy and won’t crash your workspace. To actually get **real** depth / normal / PBR / semantic maps:

1. Copy or port the relevant code from the OmniX repo (`run_pano_perception.py` and helpers) into `omni_x_utils.py`.
2. Replace the placeholder tensor generation inside each node’s `run` method with calls into your `omni_x_utils` functions.
3. Validate your results against the original OmniX scripts using a fixed test pano and comparing outputs.

You can follow the implementation plan in `design_doc_omnix_comfyui.md` for more structure.

---

## 7. License / Credits

- OmniX research and original code belong to the OmniX / HKU-MMLab authors.
- This ComfyUI node pack is just a thin integration layer.
- FLUX belongs to Black Forest Labs.

Be sure to respect all licenses and usage restrictions for the underlying models and code.
