"""
Cross-LoRA Module for OmniX

Implements multi-adapter LoRA system that allows multiple task-specific adapters
to run simultaneously on Flux transformer blocks. Based on HKU-MMLab/OmniX implementation.

This enables cross-modal conditioning where different LoRA adapters handle different
perception tasks (depth, normal, albedo, PBR) within the same Flux model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
import comfy.model_management as model_management


class CrossLoRALinear(nn.Module):
    """
    Linear layer with multiple LoRA adapters that can be dynamically switched per batch element.

    This allows heterogeneous batch processing where each sample can use a different adapter.
    """

    def __init__(self, base_linear: nn.Linear):
        super().__init__()
        self.base_linear = base_linear
        self.lora_A = nn.ModuleDict()  # Down-projection matrices
        self.lora_B = nn.ModuleDict()  # Up-projection matrices
        self.lora_scale = {}  # Scaling factors per adapter
        self.active_adapters = []

    def add_adapter(self, name: str, rank: int = 16, scale: float = 1.0):
        """Add a new LoRA adapter"""
        in_features = self.base_linear.in_features
        out_features = self.base_linear.out_features

        # Initialize LoRA matrices
        lora_A = nn.Linear(in_features, rank, bias=False)
        lora_B = nn.Linear(rank, out_features, bias=False)

        # Initialize with small random values
        nn.init.kaiming_uniform_(lora_A.weight, a=5**0.5)
        nn.init.zeros_(lora_B.weight)

        self.lora_A[name] = lora_A
        self.lora_B[name] = lora_B
        self.lora_scale[name] = scale

    def load_adapter_weights(self, name: str, state_dict: Dict[str, torch.Tensor]):
        """Load pre-trained LoRA weights from state dict"""
        if name not in self.lora_A:
            raise ValueError(f"Adapter {name} not found. Call add_adapter() first.")

        # Load weights - assumes keys like "lora_A.weight" and "lora_B.weight"
        if 'lora_A.weight' in state_dict:
            self.lora_A[name].weight.data = state_dict['lora_A.weight']
        if 'lora_B.weight' in state_dict:
            self.lora_B[name].weight.data = state_dict['lora_B.weight']

    def forward(self, x: torch.Tensor, active_adapter: Optional[str] = None) -> torch.Tensor:
        """
        Forward pass with optional LoRA adaptation.

        Args:
            x: Input tensor
            active_adapter: Name of adapter to use. If None, uses base linear only.

        Returns:
            Output tensor
        """
        # Base linear transformation
        result = self.base_linear(x)

        # Apply LoRA if adapter is specified
        if active_adapter is not None and active_adapter in self.lora_A:
            lora_out = self.lora_B[active_adapter](self.lora_A[active_adapter](x))
            result = result + lora_out * self.lora_scale[active_adapter]

        return result


def inject_cross_lora_into_model(
    model: Any,
    adapter_configs: Dict[str, Dict],
    adapter_weights: Dict[str, Dict[str, torch.Tensor]],
    device: torch.device = None
):
    """
    Inject Cross-LoRA adapters into a Flux model's transformer blocks.

    Args:
        model: ComfyUI Flux model
        adapter_configs: Dict mapping adapter names to their configurations
            Example: {"depth": {"rank": 16, "scale": 1.0, "targets": ["to_q", "to_k"]}}
        adapter_weights: Dict mapping adapter names to their state dicts
        device: Target device for adapters

    This modifies the model in-place, adding LoRA adapters to specified layers.
    """
    if device is None:
        device = model_management.get_torch_device()

    print(f"[Cross-LoRA] Injecting adapters into Flux model on {device}")

    # Access the Flux transformer
    # ComfyUI structure: model.diffusion_model
    if hasattr(model, 'diffusion_model'):
        transformer = model.diffusion_model
    elif hasattr(model, 'model'):
        transformer = model.model
    else:
        transformer = model

    # Find and patch attention layers
    # Flux typically has structure like: transformer.transformer_blocks[i].attn
    patched_count = 0

    def patch_module(module, prefix=""):
        nonlocal patched_count

        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # Target linear layers in attention modules
            # Common targets: to_q, to_k, to_v, to_out
            if isinstance(child, nn.Linear) and any(target in full_name for target in ["to_q", "to_k", "to_v", "to_out"]):
                # Replace with CrossLoRALinear
                cross_lora = CrossLoRALinear(child)

                # Add all configured adapters
                for adapter_name, config in adapter_configs.items():
                    rank = config.get('rank', 16)
                    scale = config.get('scale', 1.0)
                    targets = config.get('targets', ['to_q', 'to_k', 'to_v'])

                    # Check if this layer should get this adapter
                    if any(t in full_name for t in targets):
                        cross_lora.add_adapter(adapter_name, rank=rank, scale=scale)

                        # Load weights if available
                        if adapter_name in adapter_weights:
                            # Find matching weights for this layer
                            layer_weights = {}
                            for key, value in adapter_weights[adapter_name].items():
                                if full_name in key:
                                    # Extract lora_A or lora_B from key
                                    layer_weights[key.split('.')[-1]] = value

                            if layer_weights:
                                cross_lora.load_adapter_weights(adapter_name, layer_weights)
                                patched_count += 1

                # Replace the original linear layer
                setattr(module, name, cross_lora.to(device))
            else:
                # Recursively patch child modules
                patch_module(child, full_name)

    patch_module(transformer)

    print(f"[Cross-LoRA] Patched {patched_count} layers with adapters")

    return model


def set_active_adapters(model: Any, adapter_name: str):
    """
    Set which adapter should be active for the next forward pass.

    Args:
        model: Model with injected Cross-LoRA adapters
        adapter_name: Name of adapter to activate
    """
    # Access transformer
    if hasattr(model, 'diffusion_model'):
        transformer = model.diffusion_model
    elif hasattr(model, 'model'):
        transformer = model.model
    else:
        transformer = model

    # Recursively set active adapter on all CrossLoRALinear layers
    def set_adapter_recursive(module):
        for child in module.children():
            if isinstance(child, CrossLoRALinear):
                child.active_adapters = [adapter_name]
            else:
                set_adapter_recursive(child)

    set_adapter_recursive(transformer)
    print(f"[Cross-LoRA] Activated adapter: {adapter_name}")


def remove_cross_lora_from_model(model: Any):
    """
    Remove all Cross-LoRA adapters and restore original linear layers.

    Args:
        model: Model with injected Cross-LoRA adapters
    """
    # Access transformer
    if hasattr(model, 'diffusion_model'):
        transformer = model.diffusion_model
    elif hasattr(model, 'model'):
        transformer = model.model
    else:
        transformer = model

    removed_count = 0

    def restore_module(module):
        nonlocal removed_count

        for name, child in list(module.named_children()):
            if isinstance(child, CrossLoRALinear):
                # Restore original linear layer
                setattr(module, name, child.base_linear)
                removed_count += 1
            else:
                restore_module(child)

    restore_module(transformer)

    print(f"[Cross-LoRA] Removed {removed_count} Cross-LoRA layers")

    return model
