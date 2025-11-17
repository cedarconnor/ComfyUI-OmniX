"""
OmniX Perception Utilities

Core utilities ported from the OmniX repository for panoramic perception.
These functions handle depth, normal, PBR, and semantic segmentation
processing for equirectangular panoramas.

Reference: https://github.com/HKU-MMLab/OmniX
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any
from PIL import Image
import torch.nn.functional as F


# ============================================================================
# VAE Encoding/Decoding Utilities (FLUX-specific)
# ============================================================================

class FluxVAEHelper:
    """
    Helper class for FLUX VAE encoding/decoding with proper scaling.

    FLUX uses specific shift and scaling factors for latent space:
    - Encoding: (latents - shift_factor) * scaling_factor
    - Decoding: (latents / scaling_factor) + shift_factor
    """

    def __init__(self, vae):
        self.vae = vae
        # FLUX.1 VAE configuration
        self.shift_factor = getattr(vae.config, 'shift_factor', 0.1159)
        self.scaling_factor = getattr(vae.config, 'scaling_factor', 0.3611)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latents with FLUX scaling.

        Args:
            images: Tensor of shape (B, C, H, W), normalized to [-1, 1]

        Returns:
            Latents of shape (B, 16, H//8, W//8)
        """
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = (latents - self.shift_factor) * self.scaling_factor
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to images with FLUX scaling.

        Args:
            latents: Tensor of shape (B, 16, H//8, W//8)

        Returns:
            Images of shape (B, C, H, W), normalized to [-1, 1]
        """
        with torch.no_grad():
            latents = (latents / self.scaling_factor) + self.shift_factor
            images = self.vae.decode(latents).sample
        return images


# ============================================================================
# Depth Processing Utilities
# ============================================================================

class DepthScaler:
    """
    Scales and normalizes depth maps using various strategies.

    Modes:
    - 'minmax': Simple min-max normalization
    - 'percentile': Use percentiles to handle outliers
    - 'multi_quantiles': Multi-quantile normalization (default in OmniX)
    """

    def __init__(self, mode: str = 'multi_quantiles'):
        self.mode = mode
        self.min_val = None
        self.max_val = None

    def scale(self, depth: np.ndarray) -> np.ndarray:
        """
        Normalize depth to [0, 1] range.

        Args:
            depth: Raw depth array (H, W) or (H, W, 1)

        Returns:
            Normalized depth in [0, 1]
        """
        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth.squeeze(-1)

        if self.mode == 'minmax':
            self.min_val = depth.min()
            self.max_val = depth.max()
        elif self.mode == 'percentile':
            self.min_val = np.percentile(depth, 2)
            self.max_val = np.percentile(depth, 98)
        elif self.mode == 'multi_quantiles':
            # Use multiple quantiles for robust scaling
            q_low = np.percentile(depth, 5)
            q_high = np.percentile(depth, 95)
            self.min_val = q_low
            self.max_val = q_high
        else:
            raise ValueError(f"Unknown scaling mode: {self.mode}")

        # Avoid division by zero
        if self.max_val - self.min_val < 1e-6:
            return np.zeros_like(depth)

        normalized = (depth - self.min_val) / (self.max_val - self.min_val)
        normalized = np.clip(normalized, 0.0, 1.0)

        return normalized

    def unscale(self, normalized_depth: np.ndarray) -> np.ndarray:
        """
        Convert normalized depth back to original scale.

        Args:
            normalized_depth: Depth in [0, 1]

        Returns:
            Depth in original scale
        """
        if self.min_val is None or self.max_val is None:
            raise ValueError("Must call scale() before unscale()")

        return normalized_depth * (self.max_val - self.min_val) + self.min_val


def colorize_depth(depth: np.ndarray, cmap: str = 'inferno') -> np.ndarray:
    """
    Colorize depth map using a colormap.

    Args:
        depth: Normalized depth array (H, W) in [0, 1]
        cmap: Colormap name ('inferno', 'viridis', 'turbo', etc.)

    Returns:
        RGB image (H, W, 3) in [0, 255]
    """
    import matplotlib.cm as cm

    # Get colormap
    colormap = cm.get_cmap(cmap)

    # Apply colormap
    colored = colormap(depth)

    # Convert to RGB (drop alpha channel)
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)

    return colored_rgb


def depth_to_disparity(depth: np.ndarray, return_mask: bool = False):
    """
    Convert depth to disparity.

    Args:
        depth: Depth array (H, W)
        return_mask: Whether to return validity mask

    Returns:
        Disparity array, optionally with validity mask
    """
    eps = 1e-6
    disparity = 1.0 / (depth + eps)

    if return_mask:
        mask = depth > eps
        return disparity, mask
    return disparity


def disparity_to_depth(disparity: np.ndarray) -> np.ndarray:
    """
    Convert disparity to depth.

    Args:
        disparity: Disparity array (H, W)

    Returns:
        Depth array
    """
    eps = 1e-6
    return 1.0 / (disparity + eps)


# ============================================================================
# Normal Processing Utilities
# ============================================================================

def generate_camera_rays(height: int, width: int, device: str = 'cpu') -> torch.Tensor:
    """
    Generate camera ray directions for equirectangular panorama.

    Each pixel in an equirectangular image corresponds to a direction in 3D space.
    This function computes the XYZ direction vector for each pixel.

    Args:
        height: Image height
        width: Image width
        device: Device to create tensor on

    Returns:
        Camera rays of shape (1, 3, H, W) with XYZ directions
    """
    # Create theta (horizontal angle) and phi (vertical angle) grids
    theta = torch.linspace(0, 2 * np.pi, width, device=device)  # 0 to 2π
    phi = torch.linspace(0, np.pi, height, device=device)  # 0 to π

    # Create meshgrid
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='xy')

    # Convert spherical coordinates to Cartesian (XYZ)
    # Note: phi=0 is top, phi=π is bottom
    x = torch.sin(phi_grid) * torch.cos(theta_grid)
    y = torch.sin(phi_grid) * torch.sin(theta_grid)
    z = torch.cos(phi_grid)

    # Stack and add batch dimension
    camray = torch.stack([x, y, z], dim=0).unsqueeze(0)

    return camray


def normalize_normals(normals: np.ndarray) -> np.ndarray:
    """
    Normalize normal vectors to unit length.

    Args:
        normals: Normal map (H, W, 3) with XYZ components

    Returns:
        Normalized normals (H, W, 3)
    """
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-6)  # Avoid division by zero
    return normals / norm


def normals_to_rgb(normals: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Convert normal map from [-1, 1] to RGB [0, 255] for visualization.

    Args:
        normals: Normal map (H, W, 3) in [-1, 1]
        normalize: Whether to normalize vectors first

    Returns:
        RGB image (H, W, 3) in [0, 255]
    """
    if normalize:
        normals = normalize_normals(normals)

    # Map from [-1, 1] to [0, 255]
    rgb = ((normals + 1.0) * 127.5).astype(np.uint8)

    return rgb


def rgb_to_normals(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB normal map [0, 255] back to normals [-1, 1].

    Args:
        rgb: RGB image (H, W, 3) in [0, 255]

    Returns:
        Normal map (H, W, 3) in [-1, 1]
    """
    normals = (rgb.astype(np.float32) / 127.5) - 1.0
    return normals


# ============================================================================
# PBR Processing Utilities
# ============================================================================

def split_pbr_channels(pbr_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split PBR output into albedo, roughness, and metallic components.

    OmniX PBR output typically has:
    - First 3 channels: Albedo (RGB)
    - Channel 3: Roughness (or first channel can be roughness)
    - Channel 4: Metallic (or second channel can be metallic)

    The exact channel mapping depends on the model output format.

    Args:
        pbr_output: Tensor of shape (B, C, H, W) where C >= 3

    Returns:
        Tuple of (albedo, roughness, metallic) tensors
    """
    if pbr_output.shape[1] >= 5:
        # Format: [R, G, B, Roughness, Metallic]
        albedo = pbr_output[:, :3, :, :]
        roughness = pbr_output[:, 3:4, :, :]
        metallic = pbr_output[:, 4:5, :, :]
    elif pbr_output.shape[1] == 3:
        # Fallback: Assume RGB output, derive PBR from luminance
        albedo = pbr_output[:, :3, :, :]
        # Use luminance as roughness proxy
        roughness = 0.299 * pbr_output[:, 0:1, :, :] + \
                   0.587 * pbr_output[:, 1:2, :, :] + \
                   0.114 * pbr_output[:, 2:3, :, :]
        # Inverse luminance as metallic proxy
        metallic = 1.0 - roughness
    else:
        raise ValueError(f"Unexpected PBR output shape: {pbr_output.shape}")

    return albedo, roughness, metallic


def normalize_pbr_map(pbr_map: torch.Tensor) -> torch.Tensor:
    """
    Normalize PBR map to [0, 1] range.

    Args:
        pbr_map: PBR component tensor (B, C, H, W)

    Returns:
        Normalized tensor in [0, 1]
    """
    min_val = pbr_map.min()
    max_val = pbr_map.max()

    if max_val - min_val < 1e-6:
        return torch.zeros_like(pbr_map)

    return (pbr_map - min_val) / (max_val - min_val)


# ============================================================================
# Semantic Segmentation Utilities
# ============================================================================

def apply_semantic_palette(semantic_ids: np.ndarray, palette: str = 'ade20k') -> np.ndarray:
    """
    Apply color palette to semantic segmentation IDs.

    Args:
        semantic_ids: Semantic class IDs (H, W) with integer values
        palette: Palette name ('ade20k', 'cityscapes', 'custom')

    Returns:
        RGB image (H, W, 3) in [0, 255]
    """
    # Define color palettes
    palettes = {
        'ade20k': _get_ade20k_palette(),
        'cityscapes': _get_cityscapes_palette(),
        'custom': _get_custom_palette(),
    }

    if palette not in palettes:
        palette = 'custom'

    color_map = palettes[palette]

    # Create RGB image
    h, w = semantic_ids.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in color_map.items():
        mask = semantic_ids == class_id
        rgb[mask] = color

    return rgb


def _get_ade20k_palette() -> Dict[int, Tuple[int, int, int]]:
    """Get ADE20K color palette (simplified)."""
    # Simplified palette with common classes
    return {
        0: (0, 0, 0),        # background
        1: (120, 120, 120),  # wall
        2: (180, 120, 120),  # building
        3: (6, 230, 230),    # sky
        4: (80, 50, 50),     # floor
        5: (4, 200, 3),      # tree
        6: (120, 120, 80),   # ceiling
        7: (140, 140, 140),  # road
        8: (204, 5, 255),    # bed
        9: (230, 230, 230),  # windowpane
        10: (4, 250, 7),     # grass
        # Add more as needed
    }


def _get_cityscapes_palette() -> Dict[int, Tuple[int, int, int]]:
    """Get Cityscapes color palette."""
    return {
        0: (128, 64, 128),   # road
        1: (244, 35, 232),   # sidewalk
        2: (70, 70, 70),     # building
        3: (102, 102, 156),  # wall
        4: (190, 153, 153),  # fence
        5: (153, 153, 153),  # pole
        6: (250, 170, 30),   # traffic light
        7: (220, 220, 0),    # traffic sign
        8: (107, 142, 35),   # vegetation
        9: (152, 251, 152),  # terrain
        10: (70, 130, 180),  # sky
        # Add more as needed
    }


def _get_custom_palette() -> Dict[int, Tuple[int, int, int]]:
    """Get custom color palette with visually distinct colors."""
    return {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
        4: (255, 255, 0),
        5: (255, 0, 255),
        6: (0, 255, 255),
        7: (128, 0, 0),
        8: (0, 128, 0),
        9: (0, 0, 128),
        10: (128, 128, 0),
        11: (128, 0, 128),
        12: (0, 128, 128),
        13: (255, 128, 0),
        14: (255, 0, 128),
        15: (128, 255, 0),
    }


# ============================================================================
# Tensor Conversion Utilities
# ============================================================================

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert PyTorch tensor to PIL Image.

    Args:
        tensor: Tensor of shape (1, C, H, W) or (C, H, W), values in [0, 1] or [-1, 1]

    Returns:
        PIL Image
    """
    # Remove batch dimension if present
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)

    # Move to CPU and convert to numpy
    array = tensor.detach().cpu().numpy()

    # Transpose from (C, H, W) to (H, W, C)
    array = np.transpose(array, (1, 2, 0))

    # Normalize to [0, 1] if needed
    if array.min() < 0:
        array = (array + 1.0) / 2.0

    # Clip and convert to uint8
    array = np.clip(array * 255, 0, 255).astype(np.uint8)

    # Handle grayscale
    if array.shape[2] == 1:
        array = array.squeeze(-1)

    return Image.fromarray(array)


def pil_to_tensor(image: Image.Image, normalize: bool = True) -> torch.Tensor:
    """
    Convert PIL Image to PyTorch tensor.

    Args:
        image: PIL Image
        normalize: Whether to normalize to [-1, 1] (True) or [0, 1] (False)

    Returns:
        Tensor of shape (1, C, H, W)
    """
    # Convert to numpy array
    array = np.array(image).astype(np.float32) / 255.0

    # Handle grayscale
    if array.ndim == 2:
        array = array[:, :, np.newaxis]

    # Transpose from (H, W, C) to (C, H, W)
    array = np.transpose(array, (2, 0, 1))

    # Convert to tensor
    tensor = torch.from_numpy(array).unsqueeze(0)

    # Normalize to [-1, 1] if requested
    if normalize:
        tensor = tensor * 2.0 - 1.0

    return tensor


def comfy_image_to_tensor(image: torch.Tensor) -> torch.Tensor:
    """
    Convert ComfyUI image format to standard PyTorch tensor format.

    ComfyUI uses (B, H, W, C) format with values in [0, 1].
    We need (B, C, H, W) format.

    Args:
        image: ComfyUI image tensor (B, H, W, C)

    Returns:
        Standard tensor (B, C, H, W) in [-1, 1]
    """
    # Permute from (B, H, W, C) to (B, C, H, W)
    tensor = image.permute(0, 3, 1, 2)

    # Normalize to [-1, 1]
    tensor = tensor * 2.0 - 1.0

    return tensor


def tensor_to_comfy_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert standard PyTorch tensor to ComfyUI image format.

    Args:
        tensor: Standard tensor (B, C, H, W) in [-1, 1] or [0, 1]

    Returns:
        ComfyUI image tensor (B, H, W, C) in [0, 1]
    """
    # Normalize to [0, 1] if needed
    if tensor.min() < 0:
        tensor = (tensor + 1.0) / 2.0

    # Clamp to valid range
    tensor = torch.clamp(tensor, 0.0, 1.0)

    # Permute from (B, C, H, W) to (B, H, W, C)
    tensor = tensor.permute(0, 2, 3, 1)

    return tensor


# ============================================================================
# Tiled Decoding with Blending (for seamless panoramas)
# ============================================================================

def blend_horizontal(left: torch.Tensor, right: torch.Tensor, blend_width: int) -> torch.Tensor:
    """
    Blend two tensors horizontally with smooth transition.

    Args:
        left: Left tensor (B, C, H, W)
        right: Right tensor (B, C, H, W)
        blend_width: Width of blending region in pixels

    Returns:
        Blended tensor
    """
    if blend_width <= 0:
        return left

    # Create blend weights (linear gradient)
    blend_weights = torch.linspace(0, 1, blend_width, device=left.device)
    blend_weights = blend_weights.view(1, 1, 1, -1)

    # Extract blend regions
    left_blend = left[:, :, :, -blend_width:]
    right_blend = right[:, :, :, :blend_width]

    # Blend
    blended_region = left_blend * (1 - blend_weights) + right_blend * blend_weights

    # Construct result
    result = torch.cat([
        left[:, :, :, :-blend_width],
        blended_region,
        right[:, :, :, blend_width:]
    ], dim=3)

    return result


def add_panorama_padding(latents: torch.Tensor, padding: int = 2) -> torch.Tensor:
    """
    Add horizontal wraparound padding for seamless panorama decoding.

    Args:
        latents: Latent tensor (B, C, H, W)
        padding: Number of pixels to pad from left edge to right edge

    Returns:
        Padded latents (B, C, H, W+padding)
    """
    # Take pixels from left edge and append to right edge
    wrap_pixels = latents[:, :, :, :padding]
    return torch.cat([latents, wrap_pixels], dim=3)


def remove_panorama_padding(image: torch.Tensor, padding: int = 2) -> torch.Tensor:
    """
    Remove horizontal wraparound padding after decoding.

    Args:
        image: Image tensor (B, C, H, W)
        padding: Number of pixels to remove from right edge

    Returns:
        Cropped image (B, C, H, W-padding*8)
    """
    # Remove pixels from right edge (account for 8x upsampling)
    crop_width = padding * 8
    return image[:, :, :, :-crop_width]
