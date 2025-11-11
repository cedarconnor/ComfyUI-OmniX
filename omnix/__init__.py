"""
OmniX Core Implementation Module

Version 0.2.0 - Phase 2 Implementation Complete
"""

__version__ = "0.2.0"

from .adapters import AdapterManager, OmniXAdapters
from .perceiver import OmniXPerceiver, PanoramaEncoder
from .model_loader import OmniXModelLoader, OmniXConfig
from .generator import OmniXPanoramaGenerator, GenerationConfig
from .error_handling import (
    OmniXError,
    AdapterWeightsNotFoundError,
    OutOfMemoryError,
    ModelCompatibilityError,
    InvalidPanoramaError
)
from .utils import (
    to_comfyui_image,
    from_comfyui_image,
    validate_panorama_aspect_ratio,
    visualize_depth_map,
    normalize_normal_map
)

__all__ = [
    'AdapterManager',
    'OmniXAdapters',
    'OmniXPerceiver',
    'PanoramaEncoder',
    'OmniXModelLoader',
    'OmniXConfig',
    'OmniXPanoramaGenerator',
    'GenerationConfig',
    'OmniXError',
    'AdapterWeightsNotFoundError',
    'OutOfMemoryError',
    'ModelCompatibilityError',
    'InvalidPanoramaError',
    'to_comfyui_image',
    'from_comfyui_image',
    'validate_panorama_aspect_ratio',
    'visualize_depth_map',
    'normalize_normal_map',
]
