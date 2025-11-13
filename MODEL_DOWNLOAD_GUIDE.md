# OmniX Model Download Guide

This guide covers how to download and verify the OmniX perception adapters required by the ComfyUI-OmniX extension. You only need to perform this once per machine.

---

## 1. Requirements

- **Flux.1-dev** checkpoint (shared with other Flux workflows)
- **Disk space**: ~32 GB for all OmniX adapters
- **Python dependencies**: `huggingface_hub` (installed automatically via `pip install -r requirements.txt`)

Default adapter location:

```
ComfyUI/models/loras/omnix/
```

Each `.safetensors` file corresponds to a specific perception task.

---

## 2. Automatic Download (Recommended)

```bash
cd ComfyUI/custom_nodes/ComfyUI-OmniX
pip install huggingface_hub
python download_models.py
```

The script downloads every OmniX adapter into `ComfyUI/models/loras/omnix/`. Re-run the script anytime new adapters are released.

---

## 3. HuggingFace CLI

If you prefer the HuggingFace CLI:

```bash
pip install huggingface_hub[cli]
huggingface-cli download KevinHuang/OmniX \
    --local-dir ComfyUI/models/loras/omnix
```

This mirrors the HuggingFace repository locally and keeps the original filenames expected by the code.

---

## 4. Manual Download

1. Visit <https://huggingface.co/KevinHuang/OmniX>
2. Download every `.safetensors` file
3. Place files inside `ComfyUI/models/loras/omnix/`

Required files:

| Adapter Type | Filename |
|--------------|----------|
| RGB generation | `text_to_pano_rgb.safetensors` |
| Distance | `rgb_to_depth_depth.safetensors` |
| Normal | `rgb_to_normal_normal.safetensors` |
| Albedo | `rgb_to_albedo_albedo.safetensors` |
| PBR (roughness/metallic) | `rgb_to_pbr_pbr.safetensors` |
| Semantic (optional) | `rgb_to_semantic_semantic.safetensors` |

---

## 5. Verification Checklist

1. Confirm files exist:
   ```
   dir ComfyUI/models/loras/omnix
   ```
2. Ensure each file is larger than 200 MB (partial downloads are common errors).
3. Restart ComfyUI so that the new adapters are discovered.

If the `OmniXLoRALoader` node still reports missing files, double-check the path in the ComfyUI console output and verify spelling/case of filenames.

---

## 6. Troubleshooting

| Problem | Fix |
|---------|-----|
| `huggingface_hub` authentication error | Run `huggingface-cli login` and retry. |
| Files download as 0 bytes | Ensure you have sufficient disk space and a stable connection. |
| Script cannot find `flux1-dev.safetensors` | Place the Flux checkpoint under `ComfyUI/models/checkpoints/`. |
| Adapter loader says "not found" | Verify the adapter directory matches `ComfyUI/models/loras/omnix/` exactly. |

Still stuck? Re-run `download_models.py` with `--list-only` to print expected filenames and compare against your disk.

---

Once the adapters are installed, both the ComfyUI-native and Diffusers-based workflows can load them instantly. Enjoy extracting high-quality perception maps with OmniX!
