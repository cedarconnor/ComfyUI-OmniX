# OmniX ComfyUI Workflow Examples

This guide shows how to build OmniX workflows in ComfyUI.

## Prerequisites

1. **FLUX.1-dev** base model installed in `ComfyUI/models/checkpoints/`
2. **OmniX LoRAs** converted and placed in `ComfyUI/models/loras/`
3. **ComfyUI-OmniX** custom nodes installed

---

## Workflow 1: Text → Panorama Generation

Generate a 360° panorama from a text prompt.

### Nodes Required

1. **Load Checkpoint** → Select `flux1-dev.safetensors`
2. **Load LoRA** → Select `OmniX_text_to_pano_comfyui.safetensors`
   - Connect model from checkpoint to LoRA model input
   - Set strength to `1.0`
3. **CLIP Text Encode (Prompt)** → Enter your panorama description
   - Example: `"a photorealistic 360° panorama of a cozy modern living room, equirectangular, interior, ultra wide, 8k, highly detailed"`
4. **CLIP Text Encode (Negative)** → Enter negative prompt
   - Example: `"blurry, distorted, low quality, artifacts"`
5. **Empty Latent Image** → Set dimensions for 2:1 aspect ratio
   - Width: `2048`
   - Height: `1024`
   - Batch size: `1`
6. **KSampler**
   - Connect model from LoRA output
   - Connect positive/negative conditioning from CLIP encoders
   - Connect latent from Empty Latent Image
   - Steps: `28` (recommended)
   - CFG: `3.5`
   - Sampler: `euler`
   - Scheduler: `normal`
7. **VAE Decode** → Connect latent from KSampler
8. **Preview Image** or **Save Image** → Connect image from VAE Decode

### Connections Flow

```
Load Checkpoint → Load LoRA → KSampler → VAE Decode → Preview Image
                      ↑             ↑
CLIP Text Encode ─────┘             │
CLIP Text Encode (Neg) ──────────────┤
Empty Latent Image ──────────────────┘
```

### Recommended Settings

- **Resolution**: 1024×2048 (1:2 ratio for equirectangular)
- **Steps**: 28-50
- **CFG Scale**: 3.5-5.0
- **Sampler**: euler or dpmpp_2m
- **Prompt tips**:
  - Include "360°", "panoramic", "equirectangular"
  - Specify interior or outdoor scene
  - Add quality modifiers: "8k", "highly detailed", "photorealistic"

---

## Workflow 2: Panorama → Perception (Depth)

Extract depth map from an existing panorama image.

### Nodes Required

1. **Load Image** → Load your panorama image
2. **VAE Encode** → Encode image to latent
3. **Load Checkpoint** → Select `flux1-dev.safetensors`
4. **Load LoRA** → Select `OmniX_rgb_to_depth_comfyui.safetensors`
   - Set strength to `1.0`
5. **Empty Latent Image** → Match your panorama dimensions
6. **KSampler**
   - Connect model from LoRA
   - Connect conditioning (can use empty text or simple prompt like "depth map")
   - Steps: `28`
   - CFG: `3.5`
7. **OmniX Pano Perception: Depth** (our custom node!)
   - Connect VAE
   - Connect samples from KSampler
   - Enable normalize: `True`
   - Enable colorize: `True` (optional)
   - Select colormap: `inferno`
8. **Preview Image** → Connect depth_img output

### Connections Flow

```
Load Image → VAE Encode (as reference for dimensions)
                ↓
Load Checkpoint → Load LoRA → KSampler → OmniX Depth → Preview Image
                      ↑             ↑           ↑
CLIP Text Encode ─────┘             │           │
Empty Latent Image ──────────────────┘           │
VAE (from checkpoint) ───────────────────────────┘
```

### Important Notes

- The **conditioning** for perception tasks should be the encoded panorama
- In practice, you'll want to use the panorama as a **conditioning image**
- The KSampler generates the perception latent
- Our custom node decodes and post-processes the output

---

## Workflow 3: Panorama → Perception (All Types)

Extract all perception outputs: depth, normal, albedo, roughness, metallic, semantic.

### Strategy

Create a **parallel fan-out** workflow:

1. Start with one panorama (Load Image → VAE Encode)
2. Create separate branches for each perception type
3. Each branch has:
   - Load LoRA (with specific perception LoRA)
   - KSampler
   - Corresponding OmniX node
   - Preview/Save Image

### Perception LoRAs and Nodes

| LoRA File | OmniX Node | Output |
|-----------|------------|--------|
| `OmniX_rgb_to_depth_comfyui.safetensors` | OmniX Pano Perception: Depth | Depth map |
| `OmniX_rgb_to_normal_comfyui.safetensors` | OmniX Pano Perception: Normal | Surface normals |
| `OmniX_rgb_to_pbr_comfyui.safetensors` | OmniX Pano Perception: PBR | Albedo, Roughness, Metallic |
| `OmniX_rgb_to_semantic_comfyui.safetensors` | OmniX Pano Perception: Semantic | Semantic segmentation |

### Example: Depth + Normal Pipeline

```
Load Image ──→ VAE Encode
                   │
                   ├──→ [Depth Branch]
                   │      Load Checkpoint → Load LoRA (depth)
                   │         → KSampler → OmniX Depth → Save Image (depth.png)
                   │
                   └──→ [Normal Branch]
                          Load Checkpoint → Load LoRA (normal)
                             → KSampler → OmniX Normal → Save Image (normal.png)
```

---

## Workflow 4: Full Pipeline (Text → Pano → Perception)

Combine Workflow 1 and Workflow 2/3 to go from text prompt all the way to perception outputs.

### High-Level Flow

```
1. Text Prompt → Generate Panorama (W1)
2. Panorama → VAE Encode
3. For each perception task:
   - Load appropriate LoRA
   - Run KSampler with panorama as condition
   - Post-process with OmniX node
   - Save output
```

### Building Tips

1. **Start with W1** to generate the panorama
2. Add a **VAE Encode** node after the panorama generation
3. **Branch out** to multiple perception paths
4. Use **Reroute** nodes to keep the workflow organized
5. Use **Note** nodes to label each section

### Typical Node Count

- Text → Pano: ~8 nodes
- Per perception type: ~6 nodes
- Full pipeline with 4 perception types: ~35 nodes

---

## Advanced Tips

### Resolution Scaling

If you want higher resolution outputs:
- Generate panorama at 1024×2048
- Use **Upscale Image** nodes before perception
- Common scales: 1.5× (1536×3072) or 2× (2048×4096)
- Note: Higher resolution = more VRAM usage

### Batch Processing

Process multiple panoramas:
1. Use **Load Image Batch** node
2. Set batch size in Empty Latent Image
3. All nodes will process the batch

### Prompt Engineering for Perception

When using perception LoRAs:
- **Depth**: "depth map", "distance", "metric depth"
- **Normal**: "surface normals", "geometric details"
- **PBR**: "material properties", "albedo roughness"
- **Semantic**: "semantic segmentation", "scene understanding"

These prompts help guide the model, though the LoRA is the primary driver.

### Saving Workflows

1. Use **Queue Prompt** to test your workflow
2. Click the gear icon → **Save**
3. Name it descriptively (e.g., "omnix_text_to_pano.json")
4. Share with others or use as templates

---

## Troubleshooting

### "LoRA not found"

- Check that LoRAs are in `ComfyUI/models/loras/`
- Verify filenames match exactly
- Restart ComfyUI after adding new LoRAs

### "OmniX nodes not found"

- Check that ComfyUI-OmniX is in `ComfyUI/custom_nodes/`
- Verify `__init__.py` exists and is correct
- Check ComfyUI console for import errors
- Restart ComfyUI

### "Out of memory"

- Reduce resolution (try 768×1536 instead of 1024×2048)
- Enable **CPU offload** if available
- Process perception tasks one at a time instead of parallel
- Close other GPU applications

### "Blank or incorrect output"

- Verify you're using the correct LoRA for each perception type
- Check that KSampler has enough steps (minimum 20, recommended 28)
- Ensure CFG scale is reasonable (3.0-5.0)
- Try a different seed

### "Seam visible in panorama"

- This is normal for equirectangular images
- OmniX uses horizontal blending to minimize seams
- You can use post-processing to blend the edges
- Or crop and stitch manually in external software

---

## Example Prompts

### Interior Scenes
- `"a photorealistic 360° panorama of a modern minimalist bedroom, equirectangular, interior, morning light, 8k, highly detailed"`
- `"a 360° HDRI panorama of a cozy library with wooden shelves, equirectangular, warm lighting, ultra wide, photorealistic"`

### Outdoor Scenes
- `"a 360° panoramic view of a serene forest path, equirectangular, outdoor, natural lighting, 8k, highly detailed"`
- `"a 360° HDRI panorama of a futuristic neon city street at night, equirectangular, cinematic lighting, volumetric fog, ultrarealistic"`

### Studio/Abstract
- `"a 360° panorama of a photography studio with gradient backdrop, equirectangular, professional lighting, clean, high resolution"`
- `"a 360° abstract gradient environment, equirectangular, smooth transitions, pastel colors, minimalist"`

---

## Next Steps

1. **Experiment** with different prompts and settings
2. **Save** your best workflows as templates
3. **Combine** with other ComfyUI nodes for creative effects
4. **Export** outputs to 3D software (Blender, Unity, Unreal)

For more information, see the main README.md
