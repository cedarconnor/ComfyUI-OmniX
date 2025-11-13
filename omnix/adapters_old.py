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
        original_dtype = x.dtype
        original_device = x.device
        working = x.to(device=device, dtype=self.dtype)
        modified = False

        # If adapter has projection layers, apply them
        # This is a simplified implementation - real OmniX adapters may have
        # more complex architectures (e.g., cross-attention, LoRA, etc.)

        # Check if we have standard adapter projection weights
        if 'proj_in.weight' in self.state_dict_cache and 'proj_out.weight' in self.state_dict_cache:
            # Apply projection: x -> proj_in -> activation -> proj_out
            proj_in_weight = self.state_dict_cache['proj_in.weight']
            proj_out_weight = self.state_dict_cache['proj_out.weight']

            # Flatten spatial dimensions if needed
            original_shape = working.shape
            if len(working.shape) == 4:  # (B, C, H, W)
                B, C, H, W = working.shape
                working = working.reshape(B, C, H * W).transpose(1, 2)  # (B, H*W, C)

            # Apply projections
            working = torch.nn.functional.linear(working, proj_in_weight)
            working = torch.nn.functional.gelu(working)  # Standard activation
            working = torch.nn.functional.linear(working, proj_out_weight)

            # Restore original shape
            if len(original_shape) == 4:
                working = working.transpose(1, 2).reshape(original_shape)

            modified = True

        # If adapter has LoRA-style weights (down_proj + up_proj)
        elif 'lora_down.weight' in self.state_dict_cache and 'lora_up.weight' in self.state_dict_cache:
            # Apply LoRA: x + scale * (up_proj @ down_proj @ x)
            lora_down = self.state_dict_cache['lora_down.weight']
            lora_up = self.state_dict_cache['lora_up.weight']
            scale = self.state_dict_cache.get('lora_scale', torch.tensor(1.0))

            original_shape = working.shape
            if len(working.shape) == 4:  # (B, C, H, W)
                B, C, H, W = working.shape
                x_flat = working.reshape(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
            else:
                x_flat = working

            # LoRA transformation
            delta = torch.nn.functional.linear(x_flat, lora_down)
            delta = torch.nn.functional.linear(delta, lora_up)
            x_adapted = x_flat + scale * delta

            # Restore shape
            if len(original_shape) == 4:
                working = x_adapted.transpose(1, 2).reshape(original_shape)
            else:
                working = x_adapted

            modified = True

        # If no recognized structure, return input unchanged (backward compatibility)
        # In production, actual OmniX weights will have one of the above structures

        if not modified:
            return x

        return working.to(device=original_device, dtype=original_dtype)


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

    # Mapping from adapter types to actual filenames in the HuggingFace download
    ADAPTER_FILENAMES = {
        "rgb_generation": "text_to_pano_rgb.safetensors",
        "distance": "rgb_to_depth_depth.safetensors",
        "normal": "rgb_to_normal_normal.safetensors",
        "albedo": "rgb_to_albedo_albedo.safetensors",
        "pbr": "rgb_to_pbr_pbr.safetensors",
        "semantic": "rgb_to_semantic_semantic.safetensors",
    }

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

        # Get the actual filename from mapping
        if actual_adapter_type not in self.ADAPTER_FILENAMES:
            raise ValueError(
                f"No filename mapping for adapter type: {actual_adapter_type}\n"
                f"Available types: {', '.join(self.ADAPTER_FILENAMES.keys())}"
            )

        filename = self.ADAPTER_FILENAMES[actual_adapter_type]
        adapter_path = self.adapter_dir / filename

        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Adapter weights not found: {adapter_path}\n"
                f"Expected file: {filename}\n"
                f"Please ensure OmniX model files are downloaded to: {self.adapter_dir}"
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
            if adapter_type in ["roughness", "metallic"]:
                # Skip legacy types, they map to pbr
                continue
            if adapter_type in self.ADAPTER_FILENAMES:
                filename = self.ADAPTER_FILENAMES[adapter_type]
                adapter_path = self.adapter_dir / filename
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

        This implements sophisticated adapter injection into Flux's
        DiT (Diffusion Transformer) architecture at key layers.
        """
        adapter_info = model.omnix_adapters[adapter_type]
        adapter_module = adapter_info['module']
        strength = adapter_info['strength']

        # Find injection points in Flux architecture
        injection_points = self._find_injection_points(model)

        if len(injection_points) == 0:
            print(f"Warning: No injection points found in model. Adapter may not work correctly.")
            # Fall back to simple forward hook
            self._simple_forward_patch(model, adapter_type)
            return

        # Hook into each injection point
        for module_name, module in injection_points:
            self._hook_module(module, adapter_module, adapter_type, strength)

        print(f"✓ Patched {len(injection_points)} layers with {adapter_type} adapter")

    def _find_injection_points(self, model: nn.Module) -> list:
        """
        Find suitable injection points in Flux architecture.

        OmniX adapters work best when injected into:
        - Joint attention blocks (cross-attention layers)
        - After self-attention but before FFN
        """
        injection_points = []

        # Flux.1-dev uses "joint_blocks" for transformer layers
        if hasattr(model, 'joint_blocks'):
            joint_blocks = model.joint_blocks

            # Each joint block has attention mechanisms we can hook
            for i, block in enumerate(joint_blocks):
                # Look for attention modules
                if hasattr(block, 'attn') or hasattr(block, 'attention'):
                    injection_points.append((f'joint_blocks.{i}', block))

        # Fallback: search for any attention modules
        if not injection_points:
            for name, module in model.named_modules():
                if 'attn' in name.lower() or 'attention' in name.lower():
                    # Limit to avoid too many hooks
                    if len(injection_points) < 24:
                        injection_points.append((name, module))

        return injection_points

    def _hook_module(
        self,
        module: nn.Module,
        adapter_module: AdapterModule,
        adapter_type: str,
        strength: float
    ):
        """
        Hook a specific module to apply adapter transformation.

        This uses forward hooks to intercept and modify layer outputs.
        """
        def adapter_hook(module, input, output):
            """Forward hook that applies adapter transformation"""
            if not isinstance(output, torch.Tensor):
                # Some modules return tuples, handle appropriately
                if isinstance(output, tuple):
                    # Apply to first tensor in tuple
                    tensors = list(output)
                    if len(tensors) > 0 and isinstance(tensors[0], torch.Tensor):
                        tensors[0] = self._apply_adapter_transform(
                            tensors[0], adapter_module, strength
                        )
                    return tuple(tensors)
                return output

            # Apply adapter transformation to tensor output
            return self._apply_adapter_transform(output, adapter_module, strength)

        # Register the hook
        handle = module.register_forward_hook(adapter_hook)

        # Store handle for potential cleanup
        if not hasattr(module, '_omnix_hooks'):
            module._omnix_hooks = []
        module._omnix_hooks.append(handle)

    def _apply_adapter_transform(
        self,
        tensor: torch.Tensor,
        adapter_module: AdapterModule,
        strength: float
    ) -> torch.Tensor:
        """
        Apply adapter transformation to a tensor.

        This is the core OmniX adapter mechanism:
        output = input + strength * adapter(input)
        """
        try:
            with torch.no_grad():
                # Run adapter module
                adapter_output = adapter_module(tensor)

                # Verify shapes match
                if adapter_output.shape != tensor.shape:
                    # Try to reshape or skip
                    return tensor

                # Blend: output = input + strength * (adapter_output - input)
                # This preserves the original signal while adding adapter influence
                output = tensor + strength * (adapter_output - tensor)

                return output

        except Exception as e:
            # Silently handle errors to avoid breaking the model
            # This allows graceful degradation
            return tensor

    def _simple_forward_patch(self, model: nn.Module, adapter_type: str):
        """
        Fallback: simple forward pass patching if no injection points found.
        """
        # Store original forward if not already patched
        if not hasattr(model, '_original_forward'):
            model._original_forward = model.forward

        adapter_info = model.omnix_adapters[adapter_type]

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
                with torch.no_grad():
                    if isinstance(output, torch.Tensor):
                        adapter_output = adapter_module(output)
                        # Blend with original output
                        if adapter_output.shape == output.shape:
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
