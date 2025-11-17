"""
Key remapping for OmniX adapters to work with Diffusers FluxPipeline.

The OmniX adapters use layer names from the original training format:
- transformer.transformer_blocks.N.attn.to_q → double blocks (MMDiT)
- transformer.single_transformer_blocks.N.attn.to_q → single blocks

Diffusers FluxPipeline uses different names:
- transformer.double_blocks.N.img_attn.to_q → double blocks
- transformer.single_blocks.N.attn.to_q → single blocks

This module remaps OmniX keys to match Diffusers naming.
"""

import logging
from typing import Dict
import torch

logger = logging.getLogger(__name__)


def remap_omnix_to_diffusers_flux(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remap OmniX adapter keys to match Diffusers FluxPipeline layer names.

    Args:
        state_dict: OmniX adapter weights with original keys

    Returns:
        State dict with keys remapped to Diffusers format

    Key transformations:
        transformer.transformer_blocks.N.attn.to_X
        → transformer.double_blocks.N.img_attn.to_X

        transformer.single_transformer_blocks.N.attn.to_X
        → transformer.single_blocks.N.attn.to_X
    """
    remapped = {}
    remap_count = 0
    unknown_keys = []

    for original_key, tensor in state_dict.items():
        new_key = original_key

        # Remap double blocks (transformer_blocks → double_blocks)
        # and add img_attn intermediate module
        if ".transformer_blocks." in original_key and ".attn.to_" in original_key:
            # Example: transformer.transformer_blocks.5.attn.to_q.lora_A.weight
            # Becomes: transformer.double_blocks.5.img_attn.to_q.lora_A.weight
            new_key = original_key.replace(
                ".transformer_blocks.",
                ".double_blocks."
            ).replace(
                ".attn.to_",
                ".img_attn.to_"
            )
            remap_count += 1

        # Remap single blocks (single_transformer_blocks → single_blocks)
        elif ".single_transformer_blocks." in original_key:
            # Example: transformer.single_transformer_blocks.12.attn.to_q.lora_A.weight
            # Becomes: transformer.single_blocks.12.attn.to_q.lora_A.weight
            new_key = original_key.replace(
                ".single_transformer_blocks.",
                ".single_blocks."
            )
            remap_count += 1

        # Check if any remapping was needed but not handled
        elif "transformer_blocks" in original_key or "single_transformer_blocks" in original_key:
            unknown_keys.append(original_key)

        remapped[new_key] = tensor

    logger.info(f"Remapped {remap_count}/{len(state_dict)} keys for Diffusers FluxPipeline")

    if unknown_keys:
        logger.warning(f"Found {len(unknown_keys)} keys that couldn't be remapped:")
        for key in unknown_keys[:5]:  # Show first 5
            logger.warning(f"  {key}")

    return remapped


def verify_key_format(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """
    Analyze adapter key format to verify compatibility.

    Returns:
        Dictionary with counts of different key patterns
    """
    patterns = {
        "transformer_blocks": 0,  # OmniX double block format (needs remap)
        "double_blocks": 0,        # Diffusers double block format (correct)
        "single_transformer_blocks": 0,  # OmniX single block format (needs remap)
        "single_blocks": 0,        # Diffusers single block format (correct)
        "unknown": 0
    }

    for key in state_dict.keys():
        if "transformer_blocks" in key and "single_transformer_blocks" not in key:
            patterns["transformer_blocks"] += 1
        elif "double_blocks" in key:
            patterns["double_blocks"] += 1
        elif "single_transformer_blocks" in key:
            patterns["single_transformer_blocks"] += 1
        elif "single_blocks" in key:
            patterns["single_blocks"] += 1
        else:
            patterns["unknown"] += 1

    return patterns


if __name__ == "__main__":
    # Test with actual adapter file
    from safetensors.torch import load_file
    from pathlib import Path

    adapter_path = Path("../models/loras/omnix/rgb_to_normal_normal.safetensors")
    if adapter_path.exists():
        print(f"Loading {adapter_path}")
        state_dict = load_file(str(adapter_path))

        print("\n=== Original Key Format ===")
        patterns = verify_key_format(state_dict)
        for pattern, count in patterns.items():
            if count > 0:
                print(f"{pattern}: {count} keys")

        print("\n=== Sample Original Keys ===")
        for key in list(state_dict.keys())[:5]:
            print(f"  {key}")

        print("\n=== Remapping ===")
        remapped = remap_omnix_to_diffusers_flux(state_dict)

        print("\n=== Remapped Key Format ===")
        patterns_new = verify_key_format(remapped)
        for pattern, count in patterns_new.items():
            if count > 0:
                print(f"{pattern}: {count} keys")

        print("\n=== Sample Remapped Keys ===")
        for key in list(remapped.keys())[:5]:
            print(f"  {key}")
    else:
        print(f"Adapter file not found: {adapter_path}")
