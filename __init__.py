"""
ComfyUI-OmniX: Diffusers-based OmniX perception nodes for ComfyUI.
"""

import logging
from pathlib import Path

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    try:
        from .nodes_diffusers import (  # type: ignore
            NODE_CLASS_MAPPINGS as DIFFUSERS_NODES,
            NODE_DISPLAY_NAME_MAPPINGS as DIFFUSERS_DISPLAY,
        )
    except ImportError:
        import sys

        sys.path.append(str(Path(__file__).resolve().parent))
        from nodes_diffusers import (  # type: ignore
            NODE_CLASS_MAPPINGS as DIFFUSERS_NODES,
            NODE_DISPLAY_NAME_MAPPINGS as DIFFUSERS_DISPLAY,
        )

    from .nodes_visualization import (  # type: ignore
        NODE_CLASS_MAPPINGS as VISUAL_NODES,
        NODE_DISPLAY_NAME_MAPPINGS as VISUAL_DISPLAY,
    )

    NODE_CLASS_MAPPINGS.update(DIFFUSERS_NODES)
    NODE_CLASS_MAPPINGS.update(VISUAL_NODES)
    NODE_DISPLAY_NAME_MAPPINGS.update(DIFFUSERS_DISPLAY)
    NODE_DISPLAY_NAME_MAPPINGS.update(VISUAL_DISPLAY)
except Exception as exc:  # pragma: no cover - allows running unit tests without ComfyUI
    logging.getLogger(__name__).warning("Diffusers nodes not registered: %s", exc)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "1.1.0"
