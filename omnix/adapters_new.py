"""
OmniX Adapter Management - Proper LoRA Implementation

Loads OmniX adapter weights and injects them as LoRA into Flux transformer blocks.
Based on the real HKU-MMLab/OmniX implementation.

OmniX uses task-specific LoRA adapters that modify Flux's attention layers to
specialize the model for different perception tasks (depth, normal, albedo, PBR).
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, List, Any
import safetensors.torch
import os
import gc
from .cross_lora import inject_cross_lora_into_model, set_active_adapters, remove_cross_lora_from_model


# Adapter configurations based on OmniX repository
# Maps adapter type to its file and configuration
ADAPTER_FILENAMES = {
    "rgb_generation": "text_to_pano_rgb.safetensors",
    "distance": "rgb_to_depth_depth.safetensors",
    "normal": "rgb_to_normal_normal.safetensors",
    "albedo": "rgb_to_albedo_albedo.safetensors",
    "roughness": "rgb_to_pbr_pbr.safetensors",  # PBR adapter handles both
    "metallic": "rgb_to_pbr_pbr.safetensors",
    "pbr": "rgb_to_pbr_pbr.safetensors",
    "semantic": "rgb_to_semantic_semantic.safetensors",
}

# LoRA configuration for each adapter type
ADAPTER_CONFIGS = {
    "rgb_generation": {
        "rank": 16,
        "scale": 1.0,
        "targets": ["to_q", "to_k", "to_v", "to_out"],
    },
    "distance": {
        "rank": 16,
        "scale": 1.0,
        "targets": ["to_q", "to_k", "to_v"],
    },
    "normal": {
        "rank": 16,
        "scale": 1.0,
        "targets": ["to_q", "to_k", "to_v"],
    },
    "albedo": {
        "rank": 16,
        "scale": 1.0,
        "targets": ["to_q", "to_k", "to_v"],
    },
    "pbr": {
        "rank": 16,
        "scale": 1.0,
        "targets": ["to_q", "to_k", "to_v"],
    },
    "semantic": {
        "rank": 16,
        "scale": 1.0,
        "targets": ["to_q", "to_k", "to_v"],
    },
}


class AdapterManager:
    """
    Manages loading and injection of OmniX adapters as LoRA into Flux models.

    This replaces the old standalone AdapterModule approach with proper LoRA injection
    into Flux's transformer blocks, matching the real OmniX implementation.
    """

    def __init__(self, adapter_dir: Path, device: torch.device = None, dtype: torch.dtype = torch.float16):
        """
        Initialize adapter manager.

        Args:
            adapter_dir: Directory containing adapter .safetensors files
            device: Device to load adapters on
            dtype: Data type for adapter weights
        """
        self.adapter_dir = Path(adapter_dir)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # Loaded adapter weights
        self.adapter_weights: Dict[str, Dict[str, torch.Tensor]] = {}

        # Track which adapters are currently injected
        self.injected_model = None
        self.injected_adapters: List[str] = []

        print(f"Initialized AdapterManager: {self.adapter_dir}")

    def load_adapter(self, adapter_type: str) -> Dict[str, torch.Tensor]:
        """
        Load adapter weights from .safetensors file.

        Args:
            adapter_type: Type of adapter (distance, normal, albedo, etc.)

        Returns:
            Dictionary of adapter weights

        Raises:
            FileNotFoundError: If adapter file not found
        """
        # Check if already loaded
        if adapter_type in self.adapter_weights:
            return self.adapter_weights[adapter_type]

        # Get filename for this adapter type
        if adapter_type not in ADAPTER_FILENAMES:
            raise ValueError(f"Unknown adapter type: {adapter_type}")

        filename = ADAPTER_FILENAMES[adapter_type]
        adapter_path = self.adapter_dir / filename

        # Check if file exists
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Adapter file not found: {adapter_path}\n"
                f"Please download OmniX adapters from HuggingFace (KevinHuang/OmniX)\n"
                f"Expected location: {self.adapter_dir}"
            )

        # Load weights
        print(f"Loading {adapter_type} adapter from {adapter_path}")
        state_dict = safetensors.torch.load_file(str(adapter_path))

        # Convert to target dtype and device
        for key in state_dict:
            state_dict[key] = state_dict[key].to(device=self.device, dtype=self.dtype)

        # Cache the weights
        self.adapter_weights[adapter_type] = state_dict

        print(f"✓ Loaded {adapter_type} adapter: {len(state_dict)} weights")

        return state_dict

    def inject_adapters_into_flux(
        self,
        flux_model: Any,
        adapter_types: List[str],
        force_reload: bool = False
    ):
        """
        Inject OmniX adapters as LoRA into a Flux model.

        This modifies the Flux model in-place, adding LoRA layers to transformer blocks.

        Args:
            flux_model: ComfyUI Flux model
            adapter_types: List of adapter types to inject
            force_reload: If True, remove existing adapters and re-inject

        Returns:
            Modified Flux model with adapters injected
        """
        # Check if we need to re-inject
        if (self.injected_model is flux_model and
            set(self.injected_adapters) == set(adapter_types) and
            not force_reload):
            print(f"[AdapterManager] Adapters already injected, skipping")
            return flux_model

        # Remove existing adapters if any
        if self.injected_model is not None:
            print(f"[AdapterManager] Removing previous adapters")
            remove_cross_lora_from_model(self.injected_model)
            self.injected_model = None
            self.injected_adapters = []

        # Load all requested adapters
        print(f"[AdapterManager] Loading adapters: {adapter_types}")
        adapter_weights_dict = {}
        adapter_configs_dict = {}

        for adapter_type in adapter_types:
            # Load weights
            weights = self.load_adapter(adapter_type)
            adapter_weights_dict[adapter_type] = weights

            # Get configuration
            if adapter_type in ADAPTER_CONFIGS:
                adapter_configs_dict[adapter_type] = ADAPTER_CONFIGS[adapter_type]
            else:
                # Use default config
                adapter_configs_dict[adapter_type] = ADAPTER_CONFIGS["distance"]

        # Inject into model
        print(f"[AdapterManager] Injecting adapters into Flux model")
        inject_cross_lora_into_model(
            flux_model,
            adapter_configs_dict,
            adapter_weights_dict,
            device=self.device
        )

        # Track injection
        self.injected_model = flux_model
        self.injected_adapters = adapter_types

        print(f"✓ Injected {len(adapter_types)} adapters into Flux model")

        return flux_model

    def set_active_adapter(self, flux_model: Any, adapter_type: str):
        """
        Set which adapter should be active for the next forward pass.

        Args:
            flux_model: Flux model with injected adapters
            adapter_type: Adapter to activate

        Raises:
            ValueError: If adapter not injected
        """
        if adapter_type not in self.injected_adapters:
            raise ValueError(
                f"Adapter '{adapter_type}' not injected. "
                f"Available adapters: {self.injected_adapters}"
            )

        set_active_adapters(flux_model, adapter_type)

    def remove_adapters(self, flux_model: Any):
        """
        Remove all injected adapters from model.

        Args:
            flux_model: Model with injected adapters

        Returns:
            Model with adapters removed
        """
        if self.injected_model is flux_model:
            remove_cross_lora_from_model(flux_model)
            self.injected_model = None
            self.injected_adapters = []
            print(f"[AdapterManager] Removed all adapters")

        return flux_model

    def get_adapter_info(self) -> Dict[str, Any]:
        """Get information about loaded adapters"""
        return {
            "adapter_dir": str(self.adapter_dir),
            "loaded_adapters": list(self.adapter_weights.keys()),
            "injected_adapters": self.injected_adapters,
            "device": str(self.device),
            "dtype": str(self.dtype),
        }


class OmniXAdapters:
    """
    Legacy compatibility wrapper for old OmniXAdapters interface.

    This maintains backward compatibility with existing nodes while using
    the new LoRA-based implementation under the hood.
    """

    def __init__(self, adapter_dir: Path, dtype: torch.dtype = torch.float16):
        """Initialize with adapter directory"""
        self.manager = AdapterManager(adapter_dir, dtype=dtype)
        self.adapter_dir = adapter_dir
        self.dtype = dtype

    def get_adapter(self, adapter_type: str) -> Dict[str, torch.Tensor]:
        """
        Get adapter weights (for compatibility).

        Note: In the new implementation, adapters are injected into the model,
        so this just returns the raw weights for inspection.
        """
        return self.manager.load_adapter(adapter_type)

    def list_available_adapters(self) -> List[str]:
        """List all available adapter types"""
        available = []
        for adapter_type, filename in ADAPTER_FILENAMES.items():
            if (self.adapter_dir / filename).exists():
                available.append(adapter_type)
        return available

    def inject_into_model(self, model: Any, adapter_type: str, strength: float = 1.0):
        """
        Inject adapter into model (compatibility method).

        For the new implementation, this is handled differently - adapters are
        injected once and then activated as needed.
        """
        # This is now a no-op since injection happens at the pipeline level
        # The strength parameter is handled via ADAPTER_CONFIGS
        pass
