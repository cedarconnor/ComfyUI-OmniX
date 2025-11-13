"""
OmniX Adapter Management - Proper LoRA Implementation

Loads OmniX adapter weights and injects them as LoRA into Flux transformer blocks.
Based on the real HKU-MMLab/OmniX implementation.

OmniX uses task-specific LoRA adapters that modify Flux's attention layers to
specialize the model for different perception tasks (depth, normal, albedo, PBR).
"""

import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, List, Any, Union
import safetensors.torch
import os
import gc

logger = logging.getLogger(__name__)
try:
    from .cross_lora import inject_cross_lora_into_model, set_active_adapters, remove_cross_lora_from_model
except ModuleNotFoundError:  # pragma: no cover - allows running unit tests without ComfyUI
    def _comfy_missing(*_, **__):
        raise RuntimeError(
            "Cross-LoRA utilities require ComfyUI. Please run inside a ComfyUI environment to inject adapters."
        )

    inject_cross_lora_into_model = _comfy_missing
    set_active_adapters = _comfy_missing
    remove_cross_lora_from_model = _comfy_missing
from .adapters_old import AdapterModule as LegacyAdapterModule

# Re-export AdapterModule for backwards compatibility
AdapterModule = LegacyAdapterModule


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
        "rank": 64,  # OmniX uses rank 64
        "scale": 1.0,
        "targets": ["to_q", "to_k", "to_v", "to_out"],
    },
    "distance": {
        "rank": 64,  # OmniX uses rank 64
        "scale": 1.0,
        "targets": ["to_q", "to_k", "to_v"],
    },
    "normal": {
        "rank": 64,  # OmniX uses rank 64
        "scale": 1.0,
        "targets": ["to_q", "to_k", "to_v"],
    },
    "albedo": {
        "rank": 64,  # OmniX uses rank 64
        "scale": 1.0,
        "targets": ["to_q", "to_k", "to_v"],
    },
    "pbr": {
        "rank": 64,  # OmniX uses rank 64
        "scale": 1.0,
        "targets": ["to_q", "to_k", "to_v"],
    },
    "semantic": {
        "rank": 64,  # OmniX uses rank 64
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

        if not self.adapter_dir.exists():
            raise FileNotFoundError(f"Adapter directory not found: {self.adapter_dir}")

        # Loaded adapter weights
        self.adapter_weights: Dict[str, Dict[str, torch.Tensor]] = {}

        # Legacy compatibility caches (old AdapterModule API)
        self._legacy_loaded_adapters: Dict[str, LegacyAdapterModule] = {}
        self._loaded_adapters = self._legacy_loaded_adapters  # attribute expected by legacy tests

        # Track which adapters are currently injected
        self.injected_model = None
        self.injected_adapters: List[str] = []

        logger.info(f"Initialized AdapterManager: {self.adapter_dir}")

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
        legacy_path = self.adapter_dir / f"{adapter_type}_adapter.safetensors"

        # Check if file exists
        if not adapter_path.exists() and legacy_path.exists():
            adapter_path = legacy_path

        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Adapter file not found: {adapter_path}\n"
                f"Please download OmniX adapters from HuggingFace (KevinHuang/OmniX)\n"
                f"Expected location: {self.adapter_dir}"
            )

        # Load weights
        logger.info(f"Loading {adapter_type} adapter from {adapter_path}")
        state_dict = safetensors.torch.load_file(str(adapter_path))

        # Convert to target dtype and device
        for key in state_dict:
            state_dict[key] = state_dict[key].to(device=self.device, dtype=self.dtype)

        # Cache the weights
        self.adapter_weights[adapter_type] = state_dict

        logger.info(f"Loaded {adapter_type} adapter: {len(state_dict)} weights")

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
            logger.debug("Adapters already injected, skipping")
            return flux_model

        # Remove existing adapters if any
        if self.injected_model is not None:
            logger.info("Removing previous adapters")
            remove_cross_lora_from_model(self.injected_model)
            self.injected_model = None
            self.injected_adapters = []

        # Load all requested adapters
        logger.info(f"Loading adapters: {adapter_types}")
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
        logger.info("Injecting adapters into Flux model")
        inject_cross_lora_into_model(
            flux_model,
            adapter_configs_dict,
            adapter_weights_dict,
            device=self.device
        )

        # Track injection
        self.injected_model = flux_model
        self.injected_adapters = adapter_types

        logger.info(f"Injected {len(adapter_types)} adapters into Flux model")

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
            logger.info("Removed all adapters")

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

    # ------------------------------------------------------------------
    # Legacy compatibility layer (AdapterModule-style API)
    # ------------------------------------------------------------------
    def get_adapter(self, adapter_type: str) -> LegacyAdapterModule:
        """
        Return an AdapterModule instance for compatibility with the legacy API.
        """
        if adapter_type in self._legacy_loaded_adapters:
            return self._legacy_loaded_adapters[adapter_type]

        weights = self.load_adapter(adapter_type)
        module = LegacyAdapterModule(weights, dtype=self.dtype)
        self._legacy_loaded_adapters[adapter_type] = module
        return module

    def list_available_adapters(self) -> List[str]:
        """List adapter types that exist on disk."""
        available = []
        for adapter_type, filename in ADAPTER_FILENAMES.items():
            if (
                (self.adapter_dir / filename).exists()
                or (self.adapter_dir / f"{adapter_type}_adapter.safetensors").exists()
            ):
                available.append(adapter_type)
        return available

    def unload_adapter(self, adapter_type: str):
        """Unload adapter weights from caches to reclaim VRAM/CPU memory."""
        self.adapter_weights.pop(adapter_type, None)
        module = self._legacy_loaded_adapters.pop(adapter_type, None)
        if module is not None:
            del module
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class OmniXAdapters:
    """
    Legacy compatibility wrapper for old OmniXAdapters interface.

    This maintains backward compatibility with existing nodes while using
    the new LoRA-based implementation under the hood.
    """

    def __init__(self, adapter_dir: Union[Path, AdapterManager], dtype: torch.dtype = torch.float16):
        """Initialize with adapter directory or an existing AdapterManager."""
        if isinstance(adapter_dir, AdapterManager):
            self.manager = adapter_dir
            self.adapter_dir = adapter_dir.adapter_dir
            self.dtype = adapter_dir.dtype
        else:
            self.manager = AdapterManager(adapter_dir, dtype=dtype)
            self.adapter_dir = Path(adapter_dir)
            self.dtype = dtype

    def get_adapter(self, adapter_type: str) -> Dict[str, torch.Tensor]:
        """
        Get adapter weights (for compatibility).

        Note: In the new implementation, adapters are injected into the model,
        so this just returns the raw weights for inspection.
        """
        return self.manager.get_adapter(adapter_type)

    def list_available_adapters(self) -> List[str]:
        """List all available adapter types"""
        return self.manager.list_available_adapters()

    # Backwards-compatible alias for tests
    def list_available(self) -> List[str]:  # pragma: no cover - legacy API
        return self.list_available_adapters()

    def inject_into_model(self, model: Any, adapter_type: str, strength: float = 1.0):
        """
        Inject adapter into model (compatibility method).

        For the new implementation, this is handled differently - adapters are
        injected once and then activated as needed.
        """
        # This is now a no-op since injection happens at the pipeline level
        # The strength parameter is handled via ADAPTER_CONFIGS
        pass
