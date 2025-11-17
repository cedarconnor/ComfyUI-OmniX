import torch
from typing import Dict, Any

# NOTE:
# This file contains **stub implementations** for the OmniX perception nodes.
# They are designed to be syntactically valid and safe to import in ComfyUI,
# but they DO NOT implement the actual OmniX perception logic yet.
#
# You are expected to:
#   - Add a separate `omni_x_utils.py` file that wraps the real OmniX code
#     (e.g., functions ported from `run_pano_perception.py`).
#   - Replace the placeholder image generation in each `run` method with
#     calls into those utilities.


class OmniX_PanoPerception_Depth:
    """
    Stub node for OmniX pano depth perception.

    Inputs:
        - model: Flux model with rgb_to_depth LoRA applied
        - pano_latent: LATENT from VAEEncode
        - normalize_output: bool
    Outputs:
        - depth_img: IMAGE (grayscale / 3-channel)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "pano_latent": ("LATENT",),
                "normalize_output": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_img",)
    FUNCTION = "run"
    CATEGORY = "OmniX/Perception"

    def run(self, model: Any, pano_latent: Dict[str, torch.Tensor], normalize_output: bool = True):
        # Placeholder implementation: generate a dummy grayscale image
        samples = pano_latent.get("samples", None)
        if samples is None:
            raise ValueError("pano_latent must be a dict with key 'samples' (standard Comfy latent dict).")

        # Latent shape: (B, C, H, W). For a fake preview, we map to (B, 1, H, W) and tile to 3 channels.
        b, c, h, w = samples.shape
        # Just use the norm of the latent as a fake depth
        depth = samples.norm(dim=1, keepdim=True)
        depth = depth / (depth.max() + 1e-6)
        depth_img = depth.repeat(1, 3, 1, 1)  # 3-channel

        return (depth_img,)


class OmniX_PanoPerception_Normal:
    """
    Stub node for OmniX pano normal perception.

    Inputs:
        - model: Flux model with rgb_to_normal LoRA applied
        - pano_latent: LATENT from VAEEncode
        - normalize_output: bool
    Outputs:
        - normal_img: IMAGE (RGB normal map)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "pano_latent": ("LATENT",),
                "normalize_output": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal_img",)
    FUNCTION = "run"
    CATEGORY = "OmniX/Perception"

    def run(self, model: Any, pano_latent: Dict[str, torch.Tensor], normalize_output: bool = True):
        samples = pano_latent.get("samples", None)
        if samples is None:
            raise ValueError("pano_latent must be a dict with key 'samples'.")

        b, c, h, w = samples.shape
        # Fake normal map: normalize latent across channel dimension and remap to 0–1
        # This is purely for visualization and should be replaced by OmniX logic.
        normals = samples[:, :3, :, :]  # take first three channels
        normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-6)
        normals = (normals * 0.5) + 0.5  # [-1, 1] -> [0, 1]

        return (normals,)


class OmniX_PanoPerception_PBR:
    """
    Stub node for OmniX pano PBR perception.

    Inputs:
        - model: Flux model with rgb_to_pbr LoRA applied
        - pano_latent: LATENT from VAEEncode
        - normalize_output: bool
    Outputs:
        - albedo_img: IMAGE (RGB)
        - roughness_img: IMAGE (single or RGB)
        - metallic_img: IMAGE (single or RGB)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "pano_latent": ("LATENT",),
                "normalize_output": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("albedo_img", "roughness_img", "metallic_img")
    FUNCTION = "run"
    CATEGORY = "OmniX/Perception"

    def run(self, model: Any, pano_latent: Dict[str, torch.Tensor], normalize_output: bool = True):
        samples = pano_latent.get("samples", None)
        if samples is None:
            raise ValueError("pano_latent must be a dict with key 'samples'.")

        b, c, h, w = samples.shape

        # Fake PBR maps: these are placeholders.
        # In a real implementation, you would call into OmniX perception code here.

        # Albedo: just a normalized subset of channels.
        albedo = samples[:, :3, :, :]
        albedo = (albedo - albedo.min()) / (albedo.max() - albedo.min() + 1e-6)

        # Roughness: use channel 0 or a derived scalar.
        scalar = samples[:, 0:1, :, :].abs()
        scalar = scalar / (scalar.max() + 1e-6)
        roughness = scalar.repeat(1, 3, 1, 1)

        # Metallic: inverse of roughness just as a visual placeholder.
        metallic = 1.0 - roughness

        return (albedo, roughness, metallic)


class OmniX_PanoPerception_Semantic:
    """
    Stub node for OmniX pano semantic perception.

    Inputs:
        - model: Flux model with rgb_to_semantic LoRA applied
        - pano_latent: LATENT from VAEEncode
    Outputs:
        - semantic_img: IMAGE (colorized segmentation)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "pano_latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("semantic_img",)
    FUNCTION = "run"
    CATEGORY = "OmniX/Perception"

    def run(self, model: Any, pano_latent: Dict[str, torch.Tensor]):
        samples = pano_latent.get("samples", None)
        if samples is None:
            raise ValueError("pano_latent must be a dict with key 'samples'.")

        b, c, h, w = samples.shape

        # Fake semantic map: bucket values into a few classes and assign colors.
        # This is purely a visualization stub; real semantics should come from OmniX logic.

        x = samples[:, 0:1, :, :]
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)

        # Three fake classes based on thresholds
        class0 = (x < 0.33).float()
        class1 = ((x >= 0.33) & (x < 0.66)).float()
        class2 = (x >= 0.66).float()

        # Assign arbitrary colors
        red = class0
        green = class1
        blue = class2

        semantic_img = torch.cat([red, green, blue], dim=1)

        return (semantic_img,)


# Optional helper for ComfyUI's automatic node discovery (if imported directly as a module)
NODE_CLASS_MAPPINGS = {
    "OmniX_PanoPerception_Depth": OmniX_PanoPerception_Depth,
    "OmniX_PanoPerception_Normal": OmniX_PanoPerception_Normal,
    "OmniX_PanoPerception_PBR": OmniX_PanoPerception_PBR,
    "OmniX_PanoPerception_Semantic": OmniX_PanoPerception_Semantic,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniX_PanoPerception_Depth": "OmniX Pano Perception: Depth (Stub)",
    "OmniX_PanoPerception_Normal": "OmniX Pano Perception: Normal (Stub)",
    "OmniX_PanoPerception_PBR": "OmniX Pano Perception: PBR (Stub)",
    "OmniX_PanoPerception_Semantic": "OmniX Pano Perception: Semantic (Stub)",
}
