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
import logging

logger = logging.getLogger(__name__)


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
        # IMPORTANT: We must preserve Q, K, V independence
        # Solution: Use block-diagonal structure to maintain all learned information

        # Stack lora_A matrices vertically: [rank*3, in_dim]
        # This preserves separate down-projections for Q, K, V
        fused_A = torch.cat([q_A, k_A, v_A], dim=0)

        # Create block-diagonal lora_B: [out_dim*3, rank*3]
        # Each block corresponds to Q, K, or V projection
        rank = q_A.shape[0]
        out_dim = q_B.shape[0]
        fused_B = torch.zeros(out_dim * 3, rank * 3, dtype=q_B.dtype, device=q_B.device)

        # Block 1: Q projection
        fused_B[0:out_dim, 0:rank] = q_B
        # Block 2: K projection
        fused_B[out_dim:2*out_dim, rank:2*rank] = k_B
        # Block 3: V projection
        fused_B[2*out_dim:3*out_dim, 2*rank:3*rank] = v_B

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
        # Use block-diagonal structure to preserve Q, K, V independence

        # Stack lora_A matrices vertically: [rank*3, in_dim]
        fused_A = torch.cat([q_A, k_A, v_A], dim=0)

        # Create block-diagonal lora_B: [out_dim*3, rank*3]
        rank = q_A.shape[0]
        out_dim = q_B.shape[0]
        fused_B = torch.zeros(out_dim * 3, rank * 3, dtype=q_B.dtype, device=q_B.device)

        fused_B[0:out_dim, 0:rank] = q_B
        fused_B[out_dim:2*out_dim, rank:2*rank] = k_B
        fused_B[2*out_dim:3*out_dim, 2*rank:3*rank] = v_B

        # Store for ComfyUI single_blocks.linear1
        # Note: This only covers the QKV part of linear1
        comfyui_key = f"single_blocks.{block_idx}.linear1"
        converted[comfyui_key] = {
            "lora_A.weight": fused_A,
            "lora_B.weight": fused_B
        }

    logger.info(f"Converted {len(converted)} LoRA layers with preserved Q/K/V structure")
    return converted


def load_and_convert_adapter(adapter_path: Path) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Load OmniX adapter from safetensors and convert to ComfyUI format.

    Args:
        adapter_path: Path to OmniX .safetensors file

    Returns:
        Converted weights ready for ComfyUI injection
    """
    logger.info(f"Loading adapter from {adapter_path}")
    state_dict = safetensors.torch.load_file(str(adapter_path))

    logger.debug(f"Original: {len(state_dict)} weights")
    converted = convert_omnix_weights_to_comfyui(state_dict)
    logger.info(f"Converted: {len(converted)} layers")

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
