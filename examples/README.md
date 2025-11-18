# OmniX ComfyUI Workflow Examples

This directory contains ready-to-use ComfyUI workflow JSON files for the OmniX node pack.

## Available Workflows

### 1. `omnix_full_pipeline.json`
**Complete end-to-end pipeline: Text → Panorama → Depth**

- **Use Case**: Generate a panorama from text and extract depth in one workflow
- **What it does**:
  1. Generates a 360° panorama from your text prompt
  2. Automatically extracts and visualizes the depth map
- **Best for**: First-time users, testing the full OmniX pipeline
- **Outputs**:
  - Panorama image
  - Colorized depth map

**How to use**:
1. Load the workflow in ComfyUI
2. Edit the positive prompt (node #3) with your scene description
3. Optionally edit the negative prompt (node #4)
4. Queue prompt and wait for results!

---

### 2. `omnix_depth_from_image.json`
**Extract depth from existing panorama**

- **Use Case**: You already have a panorama and want to extract depth
- **What it does**:
  1. Loads your panorama image
  2. Extracts and colorizes the depth map
- **Best for**: Processing existing panoramas, batch depth extraction
- **Outputs**: Colorized depth map

**How to use**:
1. Load the workflow in ComfyUI
2. In the LoadImage node (node #1), select your panorama
3. Adjust colormap if desired (node #8)
4. Queue prompt

**Colormap Options**:
- `inferno` (default): Orange/yellow gradient, great for depth
- `viridis`: Blue/green/yellow, good for scientific visualization
- `plasma`: Purple/pink/yellow, high contrast
- `turbo`: Rainbow gradient, maximum color variation
- `gray`: Simple grayscale

---

### 3. `omnix_all_perceptions.json`
**Extract ALL perception types from one panorama**

- **Use Case**: Comprehensive analysis of a panorama
- **What it does**:
  1. Loads your panorama image
  2. Extracts depth (colorized)
  3. Extracts surface normals
  4. Extracts PBR materials (albedo, roughness, metallic)
- **Best for**: 3D asset creation, game development, complete scene understanding
- **Outputs**:
  - Depth map (colorized)
  - Normal map (RGB)
  - Albedo map
  - Roughness map
  - Metallic map

**How to use**:
1. Load the workflow in ComfyUI
2. In the LoadImage node (node #1), select your panorama
3. Queue prompt and wait (will take longer as it processes multiple perceptions)
4. Find all outputs in your ComfyUI output directory

**Note**: This workflow runs three separate perception pipelines in parallel. It will use more VRAM and take longer than single-perception workflows.

---

## General Tips

### Preparing Your Panorama

For best results, your input panorama should:
- Be in **equirectangular format** (2:1 aspect ratio)
- Common resolutions: 1024×2048, 1536×3072, 2048×4096
- Be well-lit and clear (avoid heavy compression artifacts)
- Represent a complete 360° view

### Workflow Customization

All workflows can be customized:

**Sampling Settings** (KSampler nodes):
- **Steps**: 28 is default, try 40-50 for higher quality
- **CFG Scale**: 3.5 is default, range 3.0-5.0
- **Seed**: Change for different variations
- **Sampler**: `euler` is default, `dpmpp_2m` also works well

**Perception Settings** (OmniX nodes):
- **Depth**: Toggle colorize on/off, try different colormaps
- **Normal**: Toggle normalize_vectors for unit-length normals
- **PBR**: Toggle normalize_maps to normalize each material map
- **Semantic**: Change palette and num_classes (not in current workflows)

### Performance Considerations

**VRAM Usage** (approximate):
- Single perception: 8-12 GB VRAM
- All perceptions (parallel): 16-24 GB VRAM

**Speed**:
- Text → Panorama: 30-60 seconds (depends on steps)
- Single perception: 20-40 seconds
- All perceptions: 60-120 seconds

If you're running low on VRAM:
1. Reduce panorama resolution (try 768×1536)
2. Enable CPU offloading in ComfyUI settings
3. Run perceptions sequentially instead of parallel
4. Close other GPU applications

### Prompt Engineering

**For Panorama Generation**:
```
Good: "a photorealistic 360° panorama of a cozy modern living room, equirectangular, interior, warm lighting, 8k, highly detailed"

Bad: "a room" (too vague, missing 360°/equirectangular keywords)
```

**For Perception Tasks**:
```
Depth: "depth map, metric depth"
Normal: "surface normals, geometric details"
PBR: "material properties, albedo roughness"
Semantic: "semantic segmentation"
```

These are optional and help guide the model, but the LoRA is the primary driver.

---

## Creating Custom Workflows

Want to create your own workflows? Here's the basic pattern:

### For Perception from Existing Image:

1. **LoadImage** → Load your panorama
2. **CheckpointLoaderSimple** → Load `flux1-dev.safetensors`
3. **LoraLoader** → Load appropriate OmniX LoRA (rgb_to_depth, rgb_to_normal, etc.)
4. **VAEEncode** → Encode panorama to latent
5. **CLIPTextEncode** × 2 → Positive and negative prompts
6. **KSampler** → Generate perception latent
7. **OmniX_PanoPerception_[Type]** → Process and visualize
8. **SaveImage** or **PreviewImage** → View results

### For Text → Panorama:

1. **CheckpointLoaderSimple** → Load `flux1-dev.safetensors`
2. **LoraLoader** → Load `OmniX_text_to_pano_comfyui.safetensors`
3. **CLIPTextEncode** × 2 → Positive and negative prompts
4. **EmptyLatentImage** → Create 2:1 latent (e.g., 2048×1024)
5. **KSampler** → Generate panorama
6. **VAEDecode** → Decode to image
7. **SaveImage** → Save panorama

Chain these patterns together for complete pipelines!

---

## Troubleshooting

### "Node not found" errors
- Ensure ComfyUI-OmniX is properly installed in `custom_nodes/`
- Restart ComfyUI completely
- Check the console for import errors

### "LoRA not found" errors
- Verify LoRAs are in `ComfyUI/models/loras/`
- Check exact filenames (case-sensitive)
- Make sure you've run the conversion script

### Blank or incorrect outputs
- Verify you're using the correct LoRA for each perception type
- Check that panorama is equirectangular format (2:1 ratio)
- Try increasing KSampler steps to 40
- Ensure CFG scale is in 3.0-5.0 range

### Out of memory
- Reduce panorama resolution
- Run one perception at a time instead of all in parallel
- Enable CPU offloading in ComfyUI settings
- Close other GPU applications

### Strange colors or artifacts
- Try different seeds
- Check input panorama quality
- Adjust colormap for depth visualization
- Increase sampling steps

---

## Additional Resources

- **[Workflow Building Guide](WORKFLOWS.md)**: Detailed guide on building workflows from scratch
- **[Main README](../README.md)**: Installation and setup instructions
- **[Design Document](../design_doc.md)**: Technical architecture details
- **[OmniX Repository](https://github.com/HKU-MMLab/OmniX)**: Original OmniX research and code

---

## Contributing Workflows

Have a cool workflow to share? We'd love to include it!

1. Test your workflow thoroughly
2. Add descriptive notes in the workflow
3. Export as JSON
4. Submit a pull request with:
   - Workflow JSON file
   - Description of what it does
   - Example images (optional but appreciated)

---

**Happy panorama generating!** 🎨🌍
