"""
OmniX Panoramic Perception Nodes for ComfyUI

These nodes implement panoramic perception capabilities from OmniX:
- Depth estimation
- Surface normal estimation
- PBR material estimation (albedo, roughness, metallic)
- Semantic segmentation

Usage:
1. Load FLUX.1-dev model
2. Apply appropriate OmniX perception LoRA
3. VAEEncode your RGB panorama → input latent (condition)
4. Use KSampler with the condition to generate perception latent
5. Connect perception latent to these nodes for post-processing

Author: Cedar Connor
Based on: https://github.com/HKU-MMLab/OmniX
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple
import folder_paths

from . import omni_x_utils as utils


class OmniX_PanoPerception_Depth:
    """
    Process depth perception output from OmniX depth estimation.

    This node takes the sampled latent from depth perception workflow,
    decodes it, and applies proper normalization and optional colorization.

    Workflow:
        VAEEncode(RGB pano) → KSampler(with rgb_to_depth LoRA)
        → This Node → Depth visualization
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "samples": ("LATENT",),
                "normalize": ("BOOLEAN", {"default": True}),
                "colorize": ("BOOLEAN", {"default": False}),
                "colormap": (["inferno", "viridis", "plasma", "turbo", "gray"], {"default": "inferno"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_img",)
    FUNCTION = "process_depth"
    CATEGORY = "OmniX/Perception"

    def process_depth(
        self,
        vae: Any,
        samples: Dict[str, torch.Tensor],
        normalize: bool = True,
        colorize: bool = False,
        colormap: str = "inferno"
    ) -> Tuple[torch.Tensor]:
        """
        Process depth perception latent into viewable depth map.

        Args:
            vae: VAE model for decoding
            samples: Latent dict from KSampler output
            normalize: Whether to normalize depth to [0, 1]
            colorize: Whether to apply colormap
            colormap: Colormap to use if colorize=True

        Returns:
            Depth visualization as IMAGE tensor (B, H, W, C)
        """
        # Get latent samples
        latent = samples["samples"]

        # Decode latents to images using FLUX VAE
        vae_helper = utils.FluxVAEHelper(vae)
        decoded = vae_helper.decode(latent)

        # Convert from (B, C, H, W) [-1, 1] to numpy
        depth_raw = decoded.detach().cpu().float()

        # Take luminance as depth (average across RGB channels)
        # Normalize to [0, 1] from [-1, 1]
        depth_raw = (depth_raw + 1.0) / 2.0
        depth = depth_raw.mean(dim=1, keepdim=True)  # (B, 1, H, W)

        # Process each image in batch
        processed_images = []
        for i in range(depth.shape[0]):
            depth_np = depth[i, 0].numpy()  # (H, W)

            # Normalize if requested
            if normalize:
                scaler = utils.DepthScaler(mode='multi_quantiles')
                depth_np = scaler.scale(depth_np)

            # Colorize if requested
            if colorize and colormap != "gray":
                depth_vis = utils.colorize_depth(depth_np, cmap=colormap)
            else:
                # Convert to grayscale RGB
                depth_vis = (depth_np * 255).astype(np.uint8)
                depth_vis = np.stack([depth_vis] * 3, axis=-1)

            # Convert to float tensor in [0, 1]
            depth_vis = torch.from_numpy(depth_vis).float() / 255.0
            processed_images.append(depth_vis)

        # Stack batch and ensure (B, H, W, C) format for ComfyUI
        output = torch.stack(processed_images, dim=0)

        return (output,)


class OmniX_PanoPerception_Normal:
    """
    Process surface normal estimation output from OmniX.

    This node takes the sampled latent from normal estimation workflow,
    decodes it, and applies normalization to ensure unit-length normal vectors.

    Workflow:
        VAEEncode(RGB pano) → KSampler(with rgb_to_normal LoRA)
        → This Node → Normal map visualization
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "samples": ("LATENT",),
                "normalize_vectors": ("BOOLEAN", {"default": True}),
                "output_format": (["rgb", "normalized"], {"default": "rgb"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal_img",)
    FUNCTION = "process_normal"
    CATEGORY = "OmniX/Perception"

    def process_normal(
        self,
        vae: Any,
        samples: Dict[str, torch.Tensor],
        normalize_vectors: bool = True,
        output_format: str = "rgb"
    ) -> Tuple[torch.Tensor]:
        """
        Process normal map latent into viewable normal visualization.

        Args:
            vae: VAE model for decoding
            samples: Latent dict from KSampler output
            normalize_vectors: Whether to normalize normal vectors to unit length
            output_format: 'rgb' for [0,1] visualization, 'normalized' for [-1,1] normals

        Returns:
            Normal map as IMAGE tensor (B, H, W, 3)
        """
        # Get latent samples
        latent = samples["samples"]

        # Decode latents to images
        vae_helper = utils.FluxVAEHelper(vae)
        decoded = vae_helper.decode(latent)

        # Convert from (B, C, H, W) to (B, H, W, C)
        normals = decoded.detach().cpu().float()

        # Process each image in batch
        processed_images = []
        for i in range(normals.shape[0]):
            # Get normal map (C, H, W) and transpose to (H, W, C)
            if normals[i].shape[0] != 3:
                raise ValueError(f"Expected 3 channels for normal map, got {normals[i].shape[0]}. Check your VAE and input latent.")
            normal_map = normals[i].permute(1, 2, 0).numpy()

            # At this point, normals are in [-1, 1] from VAE decode

            if normalize_vectors:
                # Normalize to unit length
                normal_map = utils.normalize_normals(normal_map)

            if output_format == "rgb":
                # Convert to RGB visualization [0, 1]
                normal_vis = (normal_map + 1.0) / 2.0
            else:
                # Keep as normalized normals in [-1, 1]
                # But ComfyUI expects [0, 1], so we still map
                normal_vis = (normal_map + 1.0) / 2.0

            # Convert to tensor
            normal_vis = torch.from_numpy(normal_vis).float()
            processed_images.append(normal_vis)

        # Stack batch
        output = torch.stack(processed_images, dim=0)

        return (output,)


class OmniX_PanoPerception_PBR:
    """
    Process PBR material estimation output from OmniX.

    This node takes the sampled latent from PBR estimation workflow,
    decodes it, and splits into albedo, roughness, and metallic components.

    Workflow:
        VAEEncode(RGB pano) → KSampler(with rgb_to_pbr LoRA)
        → This Node → Albedo, Roughness, Metallic maps
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "samples": ("LATENT",),
                "normalize_maps": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("albedo_img", "roughness_img", "metallic_img")
    FUNCTION = "process_pbr"
    CATEGORY = "OmniX/Perception"

    def process_pbr(
        self,
        vae: Any,
        samples: Dict[str, torch.Tensor],
        normalize_maps: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process PBR latent into albedo, roughness, and metallic maps.

        Args:
            vae: VAE model for decoding
            samples: Latent dict from KSampler output
            normalize_maps: Whether to normalize each map to [0, 1]

        Returns:
            Tuple of (albedo, roughness, metallic) as IMAGE tensors
        """
        # Get latent samples
        latent = samples["samples"]

        # Decode latents to images
        vae_helper = utils.FluxVAEHelper(vae)
        decoded = vae_helper.decode(latent)

        # decoded is (B, C, H, W) in [-1, 1]
        # Normalize to [0, 1]
        decoded = (decoded + 1.0) / 2.0

        # Split into PBR components
        # Note: The exact channel mapping depends on how the LoRA was trained
        # Common formats:
        # - Option A: RGB channels = albedo, encode roughness/metallic separately
        # - Option B: Channels 0-2 = albedo, 3 = roughness, 4 = metallic

        # We'll use a heuristic: if 3 channels, derive PBR from RGB
        # If more channels, split them appropriately

        albedo_list = []
        roughness_list = []
        metallic_list = []

        for i in range(decoded.shape[0]):
            img = decoded[i]  # (C, H, W)

            if img.shape[0] >= 5:
                # Format: [R, G, B, Roughness, Metallic, ...]
                albedo = img[:3]  # (3, H, W)
                roughness = img[3:4]  # (1, H, W)
                metallic = img[4:5]  # (1, H, W)
            elif img.shape[0] == 3:
                # Only RGB - use as albedo and derive roughness/metallic
                albedo = img[:3]
                # Use luminance for roughness
                luminance = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
                roughness = luminance.unsqueeze(0)
                # Inverse luminance for metallic
                metallic = (1.0 - luminance).unsqueeze(0)
            else:
                raise ValueError(f"Unexpected number of channels: {img.shape[0]}")

            # Normalize if requested
            if normalize_maps:
                roughness = utils.normalize_pbr_map(roughness)
                metallic = utils.normalize_pbr_map(metallic)

            # Convert to (H, W, C) format
            albedo = albedo.permute(1, 2, 0)
            roughness = roughness.permute(1, 2, 0).repeat(1, 1, 3)  # Replicate to RGB
            metallic = metallic.permute(1, 2, 0).repeat(1, 1, 3)  # Replicate to RGB

            albedo_list.append(albedo)
            roughness_list.append(roughness)
            metallic_list.append(metallic)

        # Stack batches
        albedo_out = torch.stack(albedo_list, dim=0)
        roughness_out = torch.stack(roughness_list, dim=0)
        metallic_out = torch.stack(metallic_list, dim=0)

        return (albedo_out, roughness_out, metallic_out)


class OmniX_PanoPerception_Semantic:
    """
    Process semantic segmentation output from OmniX.

    This node takes the sampled latent from semantic segmentation workflow,
    decodes it, and applies color palette for visualization.

    Workflow:
        VAEEncode(RGB pano) → KSampler(with rgb_to_semantic LoRA)
        → This Node → Colorized semantic map
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "samples": ("LATENT",),
                "palette": (["ade20k", "cityscapes", "custom"], {"default": "custom"}),
                "num_classes": ("INT", {"default": 16, "min": 2, "max": 256}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("semantic_img",)
    FUNCTION = "process_semantic"
    CATEGORY = "OmniX/Perception"

    def process_semantic(
        self,
        vae: Any,
        samples: Dict[str, torch.Tensor],
        palette: str = "custom",
        num_classes: int = 16
    ) -> Tuple[torch.Tensor]:
        """
        Process semantic segmentation latent into colorized visualization.

        Args:
            vae: VAE model for decoding
            samples: Latent dict from KSampler output
            palette: Color palette to use
            num_classes: Number of semantic classes

        Returns:
            Colorized semantic map as IMAGE tensor (B, H, W, 3)
        """
        # Get latent samples
        latent = samples["samples"]

        # Decode latents to images
        vae_helper = utils.FluxVAEHelper(vae)
        decoded = vae_helper.decode(latent)

        # Convert from (B, C, H, W) to numpy
        semantic_raw = decoded.detach().cpu().float()

        # Process each image in batch
        processed_images = []
        for i in range(semantic_raw.shape[0]):
            # Get semantic map
            sem = semantic_raw[i]  # (C, H, W)

            # Convert to class IDs
            # Method 1: Use argmax across channels if multi-channel
            # Method 2: Quantize single channel into bins

            if sem.shape[0] > 1:
                # Multi-channel: use argmax
                class_ids = torch.argmax(sem, dim=0).numpy()  # (H, W)
            else:
                # Single channel: quantize into bins
                sem_channel = sem[0]  # (H, W)
                # Normalize to [0, 1]
                sem_norm = (sem_channel + 1.0) / 2.0
                # Quantize into num_classes bins
                class_ids = (sem_norm * num_classes).long().clamp(0, num_classes - 1).numpy()

            # Apply color palette
            colored = utils.apply_semantic_palette(class_ids, palette=palette)

            # Convert to float tensor in [0, 1]
            colored = torch.from_numpy(colored).float() / 255.0
            processed_images.append(colored)

        # Stack batch
        output = torch.stack(processed_images, dim=0)

        return (output,)


# Node registration
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
