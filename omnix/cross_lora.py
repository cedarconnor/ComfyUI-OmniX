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
from .weight_converter import convert_omnix_weights_to_comfyui


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

        # Initialize LoRA matrices with compatible dtype
        base_dtype = self.base_linear.weight.dtype

        # Handle fp8 quantization: fall back to bfloat16 since fp8 doesn't support initialization
        if base_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            lora_dtype = torch.bfloat16
            print(f"[Cross-LoRA] Base layer is {base_dtype}, using bfloat16 for LoRA")
        else:
            lora_dtype = base_dtype

        lora_A = nn.Linear(in_features, rank, bias=False, dtype=lora_dtype)
        lora_B = nn.Linear(rank, out_features, bias=False, dtype=lora_dtype)

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
            active_adapter: Name of adapter to use. If None, uses self.active_adapters.

        Returns:
            Output tensor
        """
        # Base linear transformation
        result = self.base_linear(x)

        # Determine which adapter to use
        # Priority: parameter > self.active_adapters > None
        adapter_to_use = active_adapter
        if adapter_to_use is None and hasattr(self, 'active_adapters') and self.active_adapters:
            adapter_to_use = self.active_adapters[0]

        # Apply LoRA if adapter is specified
        if adapter_to_use is not None and adapter_to_use in self.lora_A:
            lora_out = self.lora_B[adapter_to_use](self.lora_A[adapter_to_use](x))
            result = result + lora_out * self.lora_scale[adapter_to_use]

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

    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"Injecting adapters into Flux model on {device}")

    # Convert OmniX weights to ComfyUI format
    logger.info(f"Converting {len(adapter_weights)} adapters to ComfyUI format...")
    converted_weights = {}
    for adapter_name, omnix_weights in adapter_weights.items():
        converted_weights[adapter_name] = convert_omnix_weights_to_comfyui(omnix_weights)

    # Access the Flux transformer
    # ComfyUI structure: model.diffusion_model or model.model.diffusion_model
    if hasattr(model, 'diffusion_model'):
        transformer = model.diffusion_model
    elif hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
        transformer = model.model.diffusion_model
    elif hasattr(model, 'model'):
        transformer = model.model
    else:
        transformer = model

    # Find and patch attention layers in Flux
    # Flux structure: double_blocks[i].img_attn, single_blocks[i].linear1/linear2
    patched_count = 0

    # Track failed adapter loads
    failed_loads = []

    # Target double_blocks - these have img_attn modules
    if hasattr(transformer, 'double_blocks'):
        logger.debug(f"Found {len(transformer.double_blocks)} double_blocks")
        for i, block in enumerate(transformer.double_blocks):
            if hasattr(block, 'img_attn'):
                img_attn = block.img_attn

                # Look for qkv or to_q/to_k/to_v inside img_attn
                for attn_name, attn_child in img_attn.named_children():
                    if isinstance(attn_child, nn.Linear) and any(x in attn_name.lower() for x in ['qkv', 'to_q', 'to_k', 'to_v']):
                        logger.debug(f"Patching double_blocks[{i}].img_attn.{attn_name}")

                        # Create CrossLoRA wrapper
                        cross_lora = CrossLoRALinear(attn_child)

                        # Add all adapters
                        for adapter_name, config in adapter_configs.items():
                            rank = config.get('rank', 16)
                            scale = config.get('scale', 1.0)
                            cross_lora.add_adapter(adapter_name, rank=rank, scale=scale)

                            # Load converted weights if available
                            layer_key = f"double_blocks.{i}.img_attn.{attn_name}"
                            if adapter_name in converted_weights and layer_key in converted_weights[adapter_name]:
                                try:
                                    cross_lora.load_adapter_weights(adapter_name, converted_weights[adapter_name][layer_key])
                                    logger.debug(f"Loaded weights for {adapter_name} in layer {layer_key}")
                                except Exception as e:
                                    error_msg = f"Failed to load {adapter_name} weights for {layer_key}: {e}"
                                    logger.error(error_msg)
                                    failed_loads.append((adapter_name, layer_key, str(e)))

                        # Replace layer
                        setattr(img_attn, attn_name, cross_lora.to(device))
                        patched_count += 1

    # Target single_blocks - these have linear1 and linear2
    if hasattr(transformer, 'single_blocks'):
        logger.debug(f"Found {len(transformer.single_blocks)} single_blocks")
        for i, block in enumerate(transformer.single_blocks):
            for linear_name in ['linear1', 'linear2']:
                if hasattr(block, linear_name):
                    linear_layer = getattr(block, linear_name)
                    if isinstance(linear_layer, nn.Linear):
                        logger.debug(f"Patching single_blocks[{i}].{linear_name}")

                        # Create CrossLoRA wrapper
                        cross_lora = CrossLoRALinear(linear_layer)

                        # Add all adapters
                        for adapter_name, config in adapter_configs.items():
                            rank = config.get('rank', 16)
                            scale = config.get('scale', 1.0)
                            cross_lora.add_adapter(adapter_name, rank=rank, scale=scale)

                            # Load converted weights if available
                            layer_key = f"single_blocks.{i}.{linear_name}"
                            if adapter_name in converted_weights and layer_key in converted_weights[adapter_name]:
                                try:
                                    cross_lora.load_adapter_weights(adapter_name, converted_weights[adapter_name][layer_key])
                                    if i == 0:  # Only log for first block to reduce verbosity
                                        logger.debug(f"Loaded weights for {adapter_name} in single_blocks")
                                except Exception as e:
                                    error_msg = f"Failed to load {adapter_name} weights for {layer_key}: {e}"
                                    logger.error(error_msg)
                                    failed_loads.append((adapter_name, layer_key, str(e)))

                        # Replace layer
                        setattr(block, linear_name, cross_lora.to(device))
                        patched_count += 1

    logger.info(f"Patched {patched_count} layers with adapters")

    # Warn if there were significant failures
    if failed_loads:
        unique_adapters = set(adapter_name for adapter_name, _, _ in failed_loads)
        logger.warning(
            f"Failed to load weights for {len(failed_loads)} layers across {len(unique_adapters)} adapters. "
            f"Affected adapters: {', '.join(unique_adapters)}. "
            "Some layers will use randomly initialized weights which may degrade quality."
        )

    return model


def set_active_adapters(model: Any, adapter_name: str):
    """
    Set which adapter should be active for the next forward pass.

    Args:
        model: Model with injected Cross-LoRA adapters
        adapter_name: Name of adapter to activate
    """
    import logging
    logger = logging.getLogger(__name__)

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
    logger.info(f"Activated adapter: {adapter_name}")


def remove_cross_lora_from_model(model: Any):
    """
    Remove all Cross-LoRA adapters and restore original linear layers.

    Args:
        model: Model with injected Cross-LoRA adapters
    """
    import logging
    logger = logging.getLogger(__name__)

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

    logger.info(f"Removed {removed_count} Cross-LoRA layers")

    return model
