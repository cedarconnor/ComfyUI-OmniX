"""
ComfyUI_OmniX - OmniX Panoramic Perception Node Pack

Integrates OmniX panoramic generation and perception capabilities
with ComfyUI using FLUX.1-dev as the base model.

Author: Cedar Connor
License: See LICENSE file
"""

from .omni_x_nodes import (
    OmniX_PanoPerception_Depth,
    OmniX_PanoPerception_Normal,
    OmniX_PanoPerception_PBR,
    OmniX_PanoPerception_Semantic,
)

# ComfyUI node registration
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

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
