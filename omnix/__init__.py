"""
OmniX Core Implementation Module

Provides core functionality for OmniX panorama generation and perception:
- Model loading (Flux + OmniX adapters)
- Panorama generation (text-to-panorama, image-to-panorama)
- Perception (depth, normals, materials extraction)
- Adapters (RGB generation, perception adapters)
- Utilities (format conversion, visualization)
"""

__version__ = "0.2.0"

from .adapters import AdapterManager, OmniXAdapters
from .perceiver import OmniXPerceiver
from .model_loader import OmniXModelLoader, load_flux_model, load_omnix_model
from .generator import OmniXPanoramaGenerator, create_generator
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

    # Model loading
    'OmniXModelLoader',
    'load_flux_model',
    'load_omnix_model',

    # Generation
    'OmniXPanoramaGenerator',
    'create_generator',

    # Perception
    'OmniXPerceiver',

    # Utilities
    'to_comfyui_image',
    'from_comfyui_image',
    'validate_panorama_aspect_ratio',
    'visualize_depth_map',
    'normalize_normal_map',
]
