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

    OmniX adapters are implemented as lightweight transformations that can be
    injected into the Flux model's layers to specialize it for different tasks.
    """

    def __init__(self, state_dict: Dict[str, torch.Tensor], dtype: torch.dtype = torch.float16):
        super().__init__()

        # Store state dict with proper handling of keys containing dots
        # PyTorch doesn't allow dots in buffer names, so we store the entire dict
        self.state_dict_cache = {}
        for key, value in state_dict.items():
            # Convert to target dtype
            value = value.to(dtype=dtype)
            # Store in our cache instead of using register_buffer with dotted names
            self.state_dict_cache[key] = value

            # For parameters that don't have dots, register them properly
            # This allows proper device movement and type conversion
            if '.' not in key:
                self.register_buffer(key, value)

        self.dtype = dtype
        self.adapter_weights = state_dict  # Keep reference for external access

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adapter transformation.

        For OmniX adapters, this typically applies a learned linear transformation
        or LoRA-style adaptation to the input features.

        Args:
            x: Input tensor (B, C, H, W) or (B, N, D)

        Returns:
            Adapted output tensor (same shape as input)
        """
        # Ensure input is on same device and dtype as adapter
        device = next(iter(self.state_dict_cache.values())).device if self.state_dict_cache else x.device
        x = x.to(device=device, dtype=self.dtype)

        # If adapter has projection layers, apply them
        # This is a simplified implementation - real OmniX adapters may have
        # more complex architectures (e.g., cross-attention, LoRA, etc.)

        # Check if we have standard adapter projection weights
        if 'proj_in.weight' in self.state_dict_cache and 'proj_out.weight' in self.state_dict_cache:
            # Apply projection: x -> proj_in -> activation -> proj_out
            proj_in_weight = self.state_dict_cache['proj_in.weight']
            proj_out_weight = self.state_dict_cache['proj_out.weight']

            # Flatten spatial dimensions if needed
            original_shape = x.shape
            if len(x.shape) == 4:  # (B, C, H, W)
                B, C, H, W = x.shape
                x = x.reshape(B, C, H * W).transpose(1, 2)  # (B, H*W, C)

            # Apply projections
            x = torch.nn.functional.linear(x, proj_in_weight)
            x = torch.nn.functional.gelu(x)  # Standard activation
            x = torch.nn.functional.linear(x, proj_out_weight)

            # Restore original shape
            if len(original_shape) == 4:
                x = x.transpose(1, 2).reshape(original_shape)

        # If adapter has LoRA-style weights (down_proj + up_proj)
        elif 'lora_down.weight' in self.state_dict_cache and 'lora_up.weight' in self.state_dict_cache:
            # Apply LoRA: x + scale * (up_proj @ down_proj @ x)
            lora_down = self.state_dict_cache['lora_down.weight']
            lora_up = self.state_dict_cache['lora_up.weight']
            scale = self.state_dict_cache.get('lora_scale', torch.tensor(1.0))

            original_shape = x.shape
            if len(x.shape) == 4:  # (B, C, H, W)
                B, C, H, W = x.shape
                x_flat = x.reshape(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
            else:
                x_flat = x

            # LoRA transformation
            delta = torch.nn.functional.linear(x_flat, lora_down)
            delta = torch.nn.functional.linear(delta, lora_up)
            x_adapted = x_flat + scale * delta

            # Restore shape
            if len(original_shape) == 4:
                x = x_adapted.transpose(1, 2).reshape(original_shape)
            else:
                x = x_adapted

        # If no recognized structure, return input unchanged (backward compatibility)
        # In production, actual OmniX weights will have one of the above structures

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
        "pbr",  # Unified PBR adapter (roughness + metallic)
        "roughness",  # Legacy: falls back to pbr
        "metallic",   # Legacy: falls back to pbr
        "semantic",
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

        # Handle legacy roughness/metallic -> pbr mapping
        actual_adapter_type = adapter_type
        if adapter_type in ["roughness", "metallic"]:
            actual_adapter_type = "pbr"
            print(f"Note: {adapter_type} uses unified PBR adapter")

        # Load from disk
        adapter_path = self.adapter_dir / f"{actual_adapter_type}_adapter.safetensors"

        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Adapter weights not found: {adapter_path}\n"
                f"Expected file: {actual_adapter_type}_adapter.safetensors\n"
                f"Please run: python download_models.py"
            )

        try:
            # Load weights using safetensors
            state_dict = safetensors.torch.load_file(str(adapter_path))

            # Create adapter module
            adapter = AdapterModule(state_dict, dtype=self.dtype)

            # Move to GPU if available
            if torch.cuda.is_available():
                adapter = adapter.cuda()

            # Cache for future use (cache under both names if legacy)
            self._loaded_adapters[actual_adapter_type] = adapter
            if adapter_type != actual_adapter_type:
                self._loaded_adapters[adapter_type] = adapter

            print(f"✓ Loaded {actual_adapter_type} adapter: {len(state_dict)} weights")

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
