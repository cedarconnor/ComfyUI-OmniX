"""
OmniX Adapter Management

Handles loading and injection of OmniX adapters into Flux models.
Adapters are separate lightweight modules for different tasks:
- RGB generation (panorama synthesis)
- Distance prediction (depth maps)
- Normal estimation
- Albedo extraction
- Roughness estimation
- Metallic estimation
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Any
import safetensors.torch
import os
import gc


class AdapterModule(nn.Module):
    """
    Wrapper for OmniX adapter weights.
    Each adapter is a lightweight module that modifies Flux's output for specific tasks.
    """

    def __init__(self, state_dict: Dict[str, torch.Tensor], dtype: torch.dtype = torch.float16):
        super().__init__()

        # Load adapter weights
        for key, value in state_dict.items():
            # Convert to target dtype
            value = value.to(dtype=dtype)
            # Register as buffer (not trainable)
            self.register_buffer(key, value)

        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adapter transformation.
        This is a placeholder - actual implementation depends on OmniX architecture.
        """
        # In practice, this would apply the adapter's learned transformation
        return x


class AdapterManager:
    """
    Manages loading and caching of OmniX adapter weights.
    Implements lazy loading to minimize memory usage.
    """

    ADAPTER_TYPES = [
        "rgb_generation",
        "distance",
        "normal",
        "albedo",
        "roughness",
        "metallic",
    ]

    def __init__(self, adapter_dir: str, dtype: torch.dtype = torch.float16):
        """
        Initialize adapter manager.

        Args:
            adapter_dir: Directory containing adapter weights (.safetensors files)
            dtype: Data type for adapter weights (fp16, fp32, bf16)
        """
        self.adapter_dir = Path(adapter_dir)
        self.dtype = dtype
        self._loaded_adapters: Dict[str, AdapterModule] = {}

        # Verify directory exists
        if not self.adapter_dir.exists():
            raise FileNotFoundError(
                f"Adapter directory not found: {self.adapter_dir}\n"
                f"Please install OmniX adapter weights to this location."
            )

        print(f"Initialized AdapterManager: {self.adapter_dir}")

    def get_adapter(self, adapter_type: str) -> AdapterModule:
        """
        Load adapter on-demand (lazy loading).

        Args:
            adapter_type: Type of adapter to load (e.g., "rgb_generation", "distance")

        Returns:
            Loaded adapter module
        """
        if adapter_type not in self.ADAPTER_TYPES:
            raise ValueError(
                f"Unknown adapter type: {adapter_type}. "
                f"Available types: {', '.join(self.ADAPTER_TYPES)}"
            )

        # Return from cache if already loaded
        if adapter_type in self._loaded_adapters:
            return self._loaded_adapters[adapter_type]

        # Load from disk
        adapter_path = self.adapter_dir / f"{adapter_type}_adapter.safetensors"

        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Adapter weights not found: {adapter_path}\n"
                f"Expected file: {adapter_type}_adapter.safetensors"
            )

        try:
            # Load weights using safetensors
            state_dict = safetensors.torch.load_file(str(adapter_path))

            # Create adapter module
            adapter = AdapterModule(state_dict, dtype=self.dtype)

            # Move to GPU if available
            if torch.cuda.is_available():
                adapter = adapter.cuda()

            # Cache for future use
            self._loaded_adapters[adapter_type] = adapter

            print(f"✓ Loaded {adapter_type} adapter: {len(state_dict)} weights")

            return adapter

        except Exception as e:
            raise RuntimeError(
                f"Failed to load adapter '{adapter_type}': {str(e)}"
            )

    def unload_adapter(self, adapter_type: str):
        """
        Unload adapter from memory to free VRAM.

        Args:
            adapter_type: Type of adapter to unload
        """
        if adapter_type in self._loaded_adapters:
            del self._loaded_adapters[adapter_type]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Unloaded {adapter_type} adapter")

    def unload_all(self):
        """Unload all adapters from memory"""
        adapter_types = list(self._loaded_adapters.keys())
        for adapter_type in adapter_types:
            self.unload_adapter(adapter_type)

    def list_available_adapters(self) -> list:
        """List adapter files available in the directory"""
        if not self.adapter_dir.exists():
            return []

        adapters = []
        for adapter_type in self.ADAPTER_TYPES:
            adapter_path = self.adapter_dir / f"{adapter_type}_adapter.safetensors"
            if adapter_path.exists():
                adapters.append(adapter_type)

        return adapters


class OmniXAdapters:
    """
    High-level interface for OmniX adapters.
    Handles adapter injection into Flux models.
    """

    def __init__(self, adapter_manager: AdapterManager):
        """
        Initialize OmniX adapters.

        Args:
            adapter_manager: Adapter manager instance
        """
        self.adapter_manager = adapter_manager

    def get_adapter(self, adapter_type: str) -> AdapterModule:
        """Get adapter module by type"""
        return self.adapter_manager.get_adapter(adapter_type)

    def inject_into_model(
        self,
        model: Any,
        adapter_type: str,
        strength: float = 1.0
    ):
        """
        Inject adapter into Flux model for inference.

        This modifies the model's forward pass to incorporate the adapter.
        The actual injection mechanism depends on the model architecture.

        Args:
            model: ComfyUI MODEL object (Flux model)
            adapter_type: Type of adapter to inject
            strength: Adapter strength multiplier (0.0 to 2.0)
        """
        adapter = self.get_adapter(adapter_type)

        # Get the actual PyTorch model from ComfyUI's MODEL wrapper
        # ComfyUI wraps models, so we need to access the underlying model
        if hasattr(model, 'model'):
            pytorch_model = model.model
        else:
            pytorch_model = model

        # Store adapter reference in model for use during forward pass
        if not hasattr(pytorch_model, 'omnix_adapters'):
            pytorch_model.omnix_adapters = {}

        pytorch_model.omnix_adapters[adapter_type] = {
            'module': adapter,
            'strength': strength
        }

        # Patch the model's forward pass
        self._patch_forward(pytorch_model, adapter_type)

        print(f"✓ Injected {adapter_type} adapter into model (strength: {strength:.2f})")

    def _patch_forward(self, model: nn.Module, adapter_type: str):
        """
        Patch model's forward pass to include adapter.

        This is a simplified implementation. The actual patching depends on:
        1. Flux model architecture (DiT structure)
        2. Where adapters should be injected (cross-attention layers)
        3. OmniX-specific adapter design

        In practice, this would:
        - Hook into specific model layers
        - Apply adapter transformation at key points
        - Blend adapter output with base model output
        """
        # Store original forward if not already patched
        if not hasattr(model, '_original_forward'):
            model._original_forward = model.forward

        # Create patched forward function
        def patched_forward(self, *args, **kwargs):
            # Call original forward
            output = self._original_forward(*args, **kwargs)

            # Apply adapter if present
            if hasattr(self, 'omnix_adapters') and adapter_type in self.omnix_adapters:
                adapter_info = self.omnix_adapters[adapter_type]
                adapter_module = adapter_info['module']
                strength = adapter_info['strength']

                # Apply adapter transformation
                # This is simplified - actual implementation would depend on:
                # - Output format (latent, image, etc.)
                # - Adapter architecture
                # - Blending strategy
                with torch.no_grad():
                    adapter_output = adapter_module(output)
                    # Blend with original output
                    output = output + strength * (adapter_output - output)

            return output

        # Replace forward method
        import types
        model.forward = types.MethodType(patched_forward, model)

    def list_available(self) -> list:
        """List available adapters"""
        return self.adapter_manager.list_available_adapters()

    def cleanup(self):
        """Cleanup all loaded adapters"""
        self.adapter_manager.unload_all()
