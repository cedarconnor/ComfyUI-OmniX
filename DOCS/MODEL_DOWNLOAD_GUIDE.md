# OmniX Model Download Guide

This guide covers how to download and verify the OmniX perception adapters required by the ComfyUI-OmniX extension. You only need to perform this once per machine.

---

## 1. Requirements

- **Flux.1-dev Diffusers config** copied manually into `ComfyUI/models/diffusers/` (see README for the exact folder structure).
- **Disk space**: ~32 GB for all OmniX adapters.
- **Adapter folder**: `ComfyUI/models/loras/omnix/` (create it if it does not exist).

Each `.safetensors` file corresponds to one perception task. No downloader or CLI tools are required—everything is copied manually.

---

## 2. Download the OmniX Adapters Manually

1. Visit <https://huggingface.co/KevinHuang/OmniX> in a browser while logged into your Hugging Face account (if access is gated).
2. Download every `.safetensors` file from the repository.
3. Copy the files into `ComfyUI/models/loras/omnix/`.

Required filenames:

| Adapter Type | Filename |
|--------------|----------|
| RGB generation | `text_to_pano_rgb.safetensors` |
| Distance | `rgb_to_depth_depth.safetensors` |
| Normal | `rgb_to_normal_normal.safetensors` |
| Albedo | `rgb_to_albedo_albedo.safetensors` |
| PBR (roughness/metallic) | `rgb_to_pbr_pbr.safetensors` |
| Semantic (optional) | `rgb_to_semantic_semantic.safetensors` |

---

## 3. Verification Checklist

1. Confirm files exist:
   ```
   dir ComfyUI/models/loras/omnix
   ```
2. Ensure each file is larger than 200 MB (partial downloads are common errors).
3. Restart ComfyUI so that the new adapters are discovered.

If the `OmniXLoRALoader` node still reports missing files, double-check the path in the ComfyUI console output and verify spelling/case of filenames.

---

## 4. Troubleshooting

| Problem | Fix |
|---------|-----|
| Files download as 0 bytes | Ensure you have sufficient disk space and a stable connection, then redownload. |
| Script cannot find `flux1-dev.safetensors` | Place the Flux checkpoint under `ComfyUI/models/checkpoints/`. |
| Adapter loader says "not found" | Verify the adapter directory matches `ComfyUI/models/loras/omnix/` exactly. |
| Diffusers loader complains about missing configs | Copy the entire `black-forest-labs/FLUX.1-dev` Diffusers repo (scheduler/text encoder/tokenizer/vae subfolders and JSON files) into `ComfyUI/models/diffusers/`. |

---

Once the adapters are installed, both the ComfyUI-native and Diffusers-based workflows can load them instantly. Enjoy extracting high-quality perception maps with OmniX!
