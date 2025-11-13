"""
Weight Converter for OmniX Adapters

Converts OmniX LoRA weights from HuggingFace Diffusers format to ComfyUI Flux format.

The key difference:
- HuggingFace: Separate to_q, to_k, to_v projections
- ComfyUI: Fused qkv projection that outputs [Q||K||V] concatenated

This module handles the weight concatenation and remapping.
"""

import torch
from typing import Dict
from pathlib import Path
import safetensors.torch


def convert_omnix_weights_to_comfyui(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Convert OmniX adapter weights from Diffusers format to ComfyUI format.

    Args:
        state_dict: Raw weights loaded from OmniX safetensors file

    Returns:
        Dictionary mapping ComfyUI layer paths to their LoRA weights
        Format: {
            "double_blocks.0.img_attn.qkv": {
                "lora_A.weight": tensor,
                "lora_B.weight": tensor
            },
            ...
        }
    """
    converted = {}

    # Process double_blocks (transformer_blocks in OmniX)
    for block_idx in range(19):  # Flux has 19 double blocks
        omnix_prefix = f"transformer.transformer_blocks.{block_idx}.attn"

        # Check if weights exist for this block
        q_A_key = f"{omnix_prefix}.to_q.lora_A.weight"
        if q_A_key not in state_dict:
            continue

        # Extract Q, K, V LoRA weights
        q_A = state_dict[f"{omnix_prefix}.to_q.lora_A.weight"]  # [rank, in_dim]
        q_B = state_dict[f"{omnix_prefix}.to_q.lora_B.weight"]  # [out_dim, rank]

        k_A = state_dict[f"{omnix_prefix}.to_k.lora_A.weight"]  # [rank, in_dim]
        k_B = state_dict[f"{omnix_prefix}.to_k.lora_B.weight"]  # [out_dim, rank]

        v_A = state_dict[f"{omnix_prefix}.to_v.lora_A.weight"]  # [rank, in_dim]
        v_B = state_dict[f"{omnix_prefix}.to_v.lora_B.weight"]  # [out_dim, rank]

        # For ComfyUI's fused qkv layer:
        # - lora_A can be averaged (same input projection)
        # - lora_B must be concatenated [Q, K, V] along output dim

        # Average the A matrices (they all project from same input space)
        fused_A = (q_A + k_A + v_A) / 3.0

        # Concatenate the B matrices along output dimension
        # Shape: [out_dim * 3, rank]
        fused_B = torch.cat([q_B, k_B, v_B], dim=0)

        # Store for ComfyUI double_blocks
        comfyui_key = f"double_blocks.{block_idx}.img_attn.qkv"
        converted[comfyui_key] = {
            "lora_A.weight": fused_A,
            "lora_B.weight": fused_B
        }

    # Process single_blocks (single_transformer_blocks in OmniX)
    for block_idx in range(38):  # Flux has 38 single blocks
        omnix_prefix = f"transformer.single_transformer_blocks.{block_idx}.attn"

        # Check if weights exist
        q_A_key = f"{omnix_prefix}.to_q.lora_A.weight"
        if q_A_key not in state_dict:
            continue

        # Extract Q, K, V LoRA weights
        q_A = state_dict[f"{omnix_prefix}.to_q.lora_A.weight"]
        q_B = state_dict[f"{omnix_prefix}.to_q.lora_B.weight"]

        k_A = state_dict[f"{omnix_prefix}.to_k.lora_A.weight"]
        k_B = state_dict[f"{omnix_prefix}.to_k.lora_B.weight"]

        v_A = state_dict[f"{omnix_prefix}.to_v.lora_A.weight"]
        v_B = state_dict[f"{omnix_prefix}.to_v.lora_B.weight"]

        # ComfyUI's linear1 fuses QKV with feedforward
        # For now, just handle QKV part (linear1's first 3*dim outputs)
        # The feedforward part is less critical for attention

        fused_A = (q_A + k_A + v_A) / 3.0
        fused_B = torch.cat([q_B, k_B, v_B], dim=0)

        # Store for ComfyUI single_blocks.linear1
        # Note: This only covers the QKV part of linear1
        comfyui_key = f"single_blocks.{block_idx}.linear1"
        converted[comfyui_key] = {
            "lora_A.weight": fused_A,
            "lora_B.weight": fused_B
        }

    print(f"[Weight Converter] Converted {len(converted)} layers")
    return converted


def load_and_convert_adapter(adapter_path: Path) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Load OmniX adapter from safetensors and convert to ComfyUI format.

    Args:
        adapter_path: Path to OmniX .safetensors file

    Returns:
        Converted weights ready for ComfyUI injection
    """
    print(f"[Weight Converter] Loading adapter from {adapter_path}")
    state_dict = safetensors.torch.load_file(str(adapter_path))

    print(f"[Weight Converter] Original: {len(state_dict)} weights")
    converted = convert_omnix_weights_to_comfyui(state_dict)
    print(f"[Weight Converter] Converted: {len(converted)} layers")

    return converted


if __name__ == "__main__":
    # Test conversion
    adapter_path = Path("C:/ComfyUI/models/loras/omnix/rgb_to_depth_depth.safetensors")
    converted = load_and_convert_adapter(adapter_path)

    # Print sample
    for key in list(converted.keys())[:3]:
        print(f"\n{key}:")
        for weight_key, weight in converted[key].items():
            print(f"  {weight_key}: {weight.shape}")
