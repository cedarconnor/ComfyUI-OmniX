"""
OmniX Utility Functions

Helper functions for image format conversion, validation, and visualization.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from PIL import Image


def to_comfyui_image(pil_image: Image.Image) -> torch.Tensor:
    """
    Convert PIL Image to ComfyUI tensor format.

    Args:
        pil_image: PIL Image (H, W, C)

    Returns:
        Tensor in ComfyUI format (B, H, W, C) with values in [0, 1]
    """
    # PIL Image -> NumPy array
    image_np = np.array(pil_image).astype(np.float32) / 255.0

    # Add batch dimension (B, H, W, C)
    image_tensor = torch.from_numpy(image_np)[None,]

    return image_tensor


def from_comfyui_image(tensor: torch.Tensor) -> Image.Image:
    """
    Convert ComfyUI tensor to PIL Image.

    Args:
        tensor: Tensor in ComfyUI format (B, H, W, C) or (H, W, C)

    Returns:
        PIL Image
    """
    # Remove batch dimension if present
    if tensor.ndim == 4:
        tensor = tensor[0]

    # Convert to numpy
    image_np = tensor.cpu().numpy()

    # Scale to 0-255 and convert to uint8
    image_np = (image_np * 255).clip(0, 255).astype(np.uint8)

    return Image.fromarray(image_np)


def validate_panorama_aspect_ratio(
    image: torch.Tensor,
    target_ratio: float = 2.0,
    tolerance: float = 0.1
) -> bool:
    """
    Validate that image has correct panorama aspect ratio.

    Args:
        image: Image tensor (B, H, W, C)
        target_ratio: Target aspect ratio (default 2.0 for equirectangular)
        tolerance: Acceptable deviation from target ratio

    Returns:
        True if valid, raises ValueError otherwise
    """
    if image.ndim != 4:
        raise ValueError(
            f"Expected 4D tensor (B, H, W, C), got shape {image.shape}"
        )

    batch, height, width, channels = image.shape

    if channels != 3 and channels != 1:
        raise ValueError(
            f"Expected 1 or 3 channels, got {channels}"
        )

    current_ratio = width / height
    ratio_diff = abs(current_ratio - target_ratio)

    if ratio_diff > tolerance:
        raise ValueError(
            f"Invalid panorama aspect ratio {current_ratio:.2f} (expected ~{target_ratio:.1f}:1)\n"
            f"Image dimensions: {width}×{height}\n"
            f"Use OmniXPanoramaValidator node to fix aspect ratio."
        )

    return True


def visualize_depth_map(depth: torch.Tensor) -> torch.Tensor:
    """
    Convert depth map to colored visualization using viridis colormap.

    Args:
        depth: Depth tensor (B, H, W, 1) with values in [0, 1]

    Returns:
        Colored depth map (B, H, W, 3) for visualization
    """
    # Ensure depth is in [0, 1] range
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    # Remove channel dimension for colormap application
    if depth_normalized.shape[-1] == 1:
        depth_normalized = depth_normalized.squeeze(-1)

    # Convert to numpy for colormap
    depth_np = depth_normalized.cpu().numpy()

    # Apply viridis colormap
    try:
        import matplotlib.cm as cm
        colormap = cm.get_cmap('viridis')
        colored = colormap(depth_np)[:, :, :, :3]  # Remove alpha channel
    except ImportError:
        # Fallback: simple grayscale if matplotlib not available
        print("Warning: matplotlib not available, using grayscale depth visualization")
        depth_np_expanded = np.stack([depth_np] * 3, axis=-1)
        colored = depth_np_expanded

    # Convert back to tensor
    colored_tensor = torch.from_numpy(colored).float()

    return colored_tensor


def normalize_normal_map(normal: torch.Tensor) -> torch.Tensor:
    """
    Normalize and convert normal map from [-1, 1] to [0, 1] for visualization.

    Args:
        normal: Normal map (B, H, W, 3) with values in [-1, 1]

    Returns:
        Normalized normal map (B, H, W, 3) with values in [0, 1]
    """
    # Ensure unit vectors
    normal = F.normalize(normal, dim=-1, p=2)

    # Convert from [-1, 1] to [0, 1] for visualization
    normal_vis = (normal + 1.0) / 2.0

    return normal_vis


def panorama_to_cubemap(
    equirect: torch.Tensor,
    face_size: int = 512
) -> Tuple[torch.Tensor, ...]:
    """
    Convert equirectangular panorama to cubemap (6 faces).

    Args:
        equirect: Equirectangular image (B, H, W, C)
        face_size: Size of each cube face

    Returns:
        Tuple of 6 faces (front, right, back, left, top, bottom)
    """
    # This is a placeholder for future implementation
    # Actual conversion requires spherical coordinate mapping
    raise NotImplementedError(
        "Cubemap conversion not yet implemented. "
        "This will be added in a future version."
    )


def pack_pbr_materials(
    albedo: torch.Tensor,
    roughness: torch.Tensor,
    metallic: torch.Tensor,
    normal: Optional[torch.Tensor] = None
) -> dict:
    """
    Pack PBR material maps into standard formats.

    Args:
        albedo: Albedo/base color (B, H, W, 3)
        roughness: Roughness map (B, H, W, 1)
        metallic: Metallic map (B, H, W, 1)
        normal: Optional normal map (B, H, W, 3)

    Returns:
        Dictionary with packed material maps
    """
    materials = {
        'albedo': albedo,
        'roughness': roughness,
        'metallic': metallic,
    }

    if normal is not None:
        materials['normal'] = normal

    # Pack roughness and metallic into ORM (Occlusion-Roughness-Metallic)
    # Format used by glTF and many engines
    # Channel R: Occlusion (we'll use white/1.0 as we don't have AO)
    # Channel G: Roughness
    # Channel B: Metallic

    batch, height, width, _ = albedo.shape

    # Create ORM texture
    occlusion = torch.ones_like(roughness)  # Full occlusion (no AO)
    orm = torch.cat([occlusion, roughness, metallic], dim=-1)

    materials['orm'] = orm

    return materials


def adaptive_batch_size(
    vram_available: int,
    resolution: Tuple[int, int],
    base_memory_gb: float = 8.0
) -> int:
    """
    Calculate optimal batch size based on available VRAM and resolution.

    Args:
        vram_available: Available VRAM in GB
        resolution: Image resolution (width, height)
        base_memory_gb: Base memory required for model (GB)

    Returns:
        Recommended batch size
    """
    width, height = resolution

    # Estimate memory per image (rough approximation)
    pixels = width * height
    memory_per_image_gb = (pixels * 4 * 3) / (1024 ** 3)  # 4 bytes per float, 3 channels

    # Account for intermediate activations (roughly 2x)
    memory_per_image_gb *= 2

    # Safety margin (20%)
    safety_margin = 0.2
    available = vram_available * (1 - safety_margin) - base_memory_gb

    if available <= 0:
        return 1  # Minimum batch size

    batch_size = max(1, int(available / memory_per_image_gb))

    return batch_size


def diagnose_memory():
    """
    Print current GPU memory usage for debugging.
    """
    import torch

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device = torch.device(f'cuda:{i}')
            props = torch.cuda.get_device_properties(device)
            total_memory = props.total_memory / 1e9

            allocated = torch.cuda.memory_allocated(device) / 1e9
            reserved = torch.cuda.memory_reserved(device) / 1e9

            print(f"GPU {i}: {props.name}")
            print(f"  Total VRAM: {total_memory:.2f} GB")
            print(f"  Allocated: {allocated:.2f} GB ({allocated/total_memory*100:.1f}%)")
            print(f"  Reserved: {reserved:.2f} GB ({reserved/total_memory*100:.1f}%)")
            print(f"  Free: {total_memory - reserved:.2f} GB")
    else:
        print("CUDA not available - running on CPU")


def cleanup_memory():
    """
    Aggressive memory cleanup to free VRAM.
    """
    import gc
    import torch

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("Memory cleanup completed")
