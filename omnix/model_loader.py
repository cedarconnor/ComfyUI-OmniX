"""
OmniX Model Loader

Handles loading and initialization of Flux models with OmniX adapters.
Integrates with ComfyUI's existing model loading infrastructure.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import os
import folder_paths
import comfy.model_management as mm
import comfy.sd
import comfy.utils

from .adapters import AdapterManager, OmniXAdapters


class OmniXModelLoader:
    """
    Loads and manages Flux.1-dev models with OmniX adapter integration.

    This class handles:
    1. Loading base Flux.1-dev model (via ComfyUI's infrastructure)
    2. Loading OmniX adapter weights
    3. Managing model/adapter lifecycle and memory
    """

    def __init__(
        self,
        flux_model: Any,  # ComfyUI MODEL object
        adapters: OmniXAdapters,
        vae: Optional[Any] = None,  # ComfyUI VAE object
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize OmniX model loader.

        Args:
            flux_model: ComfyUI Flux model object
            adapters: OmniX adapters instance
            vae: Optional VAE for encoding/decoding (uses model's VAE if None)
            dtype: Data type for inference
        """
        self.flux_model = flux_model
        self.adapters = adapters
        self.vae = vae
        self.dtype = dtype

        # Track which adapters are currently active
        self.active_adapters: Dict[str, float] = {}

    def apply_adapter(self, adapter_type: str, strength: float = 1.0):
        """
        Apply an adapter to the model.

        Args:
            adapter_type: Type of adapter to apply
            strength: Adapter strength (0.0 to 2.0)
        """
        # Inject adapter into model
        self.adapters.inject_into_model(
            self.flux_model,
            adapter_type,
            strength=strength
        )

        # Track active adapter
        self.active_adapters[adapter_type] = strength

        print(f"✓ Applied {adapter_type} adapter (strength: {strength:.2f})")

    def remove_adapter(self, adapter_type: str):
        """
        Remove an adapter from the model.

        Args:
            adapter_type: Type of adapter to remove
        """
        # Get the underlying model
        pytorch_model = self.flux_model.model if hasattr(self.flux_model, 'model') else self.flux_model

        # Remove adapter if present
        if hasattr(pytorch_model, 'omnix_adapters') and adapter_type in pytorch_model.omnix_adapters:
            del pytorch_model.omnix_adapters[adapter_type]

        # Remove from tracking
        if adapter_type in self.active_adapters:
            del self.active_adapters[adapter_type]

        print(f"✓ Removed {adapter_type} adapter")

    def get_model(self) -> Any:
        """Get the Flux model (with adapters applied)"""
        return self.flux_model

    def get_vae(self) -> Any:
        """Get the VAE"""
        return self.vae

    def get_adapters(self) -> OmniXAdapters:
        """Get the adapters instance"""
        return self.adapters

    def cleanup(self):
        """Cleanup resources"""
        # Remove all adapters
        for adapter_type in list(self.active_adapters.keys()):
            self.remove_adapter(adapter_type)

        # Cleanup adapter manager
        self.adapters.cleanup()

        print("✓ Cleaned up OmniX model loader")

    @classmethod
    def from_comfyui(
        cls,
        model: Any,  # ComfyUI MODEL
        adapter_path: str,
        vae: Optional[Any] = None,
        dtype: torch.dtype = torch.float16
    ) -> 'OmniXModelLoader':
        """
        Create OmniXModelLoader from ComfyUI model and adapter path.

        Args:
            model: ComfyUI MODEL object (Flux model)
            adapter_path: Path to OmniX adapter directory
            vae: Optional ComfyUI VAE object
            dtype: Data type for adapters

        Returns:
            OmniXModelLoader instance
        """
        # Load adapters
        adapter_manager = AdapterManager(adapter_path, dtype=dtype)
        adapters = OmniXAdapters(adapter_manager)

        # Create loader
        loader = cls(
            flux_model=model,
            adapters=adapters,
            vae=vae,
            dtype=dtype
        )

        print(f"✓ Initialized OmniX model loader")
        print(f"  Model: Flux.1-dev")
        print(f"  Adapters: {adapter_path}")
        print(f"  Precision: {dtype}")

        return loader


def load_flux_model(
    checkpoint_name: str,
    device: Optional[torch.device] = None
) -> Tuple[Any, Any, Any]:
    """
    Load Flux.1-dev model using ComfyUI's infrastructure.

    Args:
        checkpoint_name: Name of checkpoint file
        device: Target device (auto-detected if None)

    Returns:
        Tuple of (model, clip, vae)
    """
    # Use ComfyUI's model loading
    checkpoint_path = folder_paths.get_full_path("checkpoints", checkpoint_name)

    if checkpoint_path is None:
        raise FileNotFoundError(
            f"Flux checkpoint not found: {checkpoint_name}\n"
            f"Please place Flux.1-dev checkpoint in ComfyUI/models/checkpoints/"
        )

    print(f"Loading Flux model from: {checkpoint_path}")

    # Load using ComfyUI's loader
    out = comfy.sd.load_checkpoint_guess_config(
        checkpoint_path,
        output_vae=True,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings")
    )

    model = out[0]  # ModelPatcher
    clip = out[1]   # CLIP
    vae = out[2]    # VAE

    print("✓ Loaded Flux model")

    return model, clip, vae


def load_omnix_model(
    flux_checkpoint: str,
    adapter_preset: str = "omnix-base",
    dtype: torch.dtype = torch.float16
) -> OmniXModelLoader:
    """
    Load complete OmniX model (Flux + adapters).

    Args:
        flux_checkpoint: Flux.1-dev checkpoint name
        adapter_preset: OmniX adapter preset name
        dtype: Data type for inference

    Returns:
        OmniXModelLoader instance ready for generation/perception
    """
    # Load Flux model
    model, clip, vae = load_flux_model(flux_checkpoint)

    # Find adapter path - check loras/omnix directory first
    loras_path = folder_paths.get_folder_paths("loras")
    if loras_path:
        # Files are directly in loras/omnix
        adapter_path = os.path.join(loras_path[0], "omnix")
    else:
        # Fallback to checking omnix directory with preset subdirectories
        adapter_base_path = folder_paths.get_folder_paths("omnix")
        if not adapter_base_path:
            import os
            adapter_base_path = [os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "models",
                "omnix"
            )]
        adapter_path = os.path.join(adapter_base_path[0], adapter_preset)

    # Create OmniX loader
    omnix_loader = OmniXModelLoader.from_comfyui(
        model=model,
        adapter_path=adapter_path,
        vae=vae,
        dtype=dtype
    )

    # Store CLIP for text encoding
    omnix_loader.clip = clip

    return omnix_loader
