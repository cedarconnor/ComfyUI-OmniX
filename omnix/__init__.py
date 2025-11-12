"""
OmniX Core Implementation Module - Perception Only

Provides core functionality for OmniX panorama perception:
- Adapter loading and LoRA injection
- Utilities (format conversion, visualization)
"""

__version__ = "0.3.0-perception"

from .adapters import AdapterManager, OmniXAdapters
from .utils import (
    to_comfyui_image,
    from_comfyui_image,
    validate_panorama_aspect_ratio,
    visualize_depth_map,
    normalize_normal_map
)

__all__ = [
    # Adapters
    'AdapterManager',
    'OmniXAdapters',

    # Utilities
    'to_comfyui_image',
    'from_comfyui_image',
    'validate_panorama_aspect_ratio',
    'visualize_depth_map',
    'normalize_normal_map',
]
