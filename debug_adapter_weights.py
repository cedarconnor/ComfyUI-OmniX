"""
Debug script to inspect OmniX adapter weight structure and verify PEFT compatibility.

This script helps diagnose why the OmniX adapters may not be producing correct outputs
by examining the weight keys and shapes.
"""

import torch
from pathlib import Path
from safetensors.torch import load_file
import sys


def inspect_adapter_weights(adapter_path: Path):
    """Inspect OmniX adapter weight structure."""
    print(f"\n{'='*80}")
    print(f"Inspecting: {adapter_path.name}")
    print(f"{'='*80}\n")

    # Load weights
    state_dict = load_file(str(adapter_path))
    print(f"Total weight tensors: {len(state_dict)}\n")

    # Group weights by layer type
    layer_types = {}
    for key in state_dict.keys():
        # Extract layer type (e.g., "transformer.transformer_blocks.0.attn.to_q.lora_A.weight")
        parts = key.split('.')
        if 'lora_A' in key or 'lora_B' in key:
            # Get the path before lora_A/lora_B
            layer_path = '.'.join(parts[:-2])  # Remove "lora_A/B.weight"
            if layer_path not in layer_types:
                layer_types[layer_path] = []
            layer_types[layer_path].append(key)

    # Print layer structure
    print(f"Found {len(layer_types)} unique LoRA layers:\n")

    # Sample some layers
    for i, (layer_path, keys) in enumerate(list(layer_types.items())[:5]):
        print(f"{i+1}. {layer_path}")
        for key in sorted(keys):
            shape = state_dict[key].shape
            print(f"   - {key.split('.')[-2]}.{key.split('.')[-1]}: {shape}")
        print()

    # Check for specific patterns
    print("\n" + "="*80)
    print("Weight Key Patterns:")
    print("="*80 + "\n")

    has_transformer_blocks = any('transformer_blocks' in k for k in state_dict.keys())
    has_single_blocks = any('single_transformer_blocks' in k for k in state_dict.keys())
    has_double_blocks = any('double_blocks' in k for k in state_dict.keys())
    has_to_q = any('to_q' in k for k in state_dict.keys())
    has_to_k = any('to_k' in k for k in state_dict.keys())
    has_to_v = any('to_v' in k for k in state_dict.keys())
    has_qkv = any('qkv' in k for k in state_dict.keys())

    print(f"Contains 'transformer_blocks' (Diffusers style): {has_transformer_blocks}")
    print(f"Contains 'single_transformer_blocks': {has_single_blocks}")
    print(f"Contains 'double_blocks' (ComfyUI style): {has_double_blocks}")
    print(f"Contains 'to_q' (separate Q/K/V): {has_to_q}")
    print(f"Contains 'to_k': {has_to_k}")
    print(f"Contains 'to_v': {has_to_v}")
    print(f"Contains 'qkv' (fused): {has_qkv}")

    # Check rank
    print("\n" + "="*80)
    print("LoRA Rank Detection:")
    print("="*80 + "\n")

    for key in state_dict.keys():
        if 'lora_A.weight' in key:
            rank = state_dict[key].shape[0]
            print(f"LoRA rank: {rank}")
            break

    # Print all keys for the first block
    print("\n" + "="*80)
    print("Complete keys for first block:")
    print("="*80 + "\n")

    first_block_keys = [k for k in sorted(state_dict.keys()) if 'blocks.0.' in k or 'blocks.1.' in k][:20]
    for key in first_block_keys:
        print(f"  {key}: {state_dict[key].shape}")

    return state_dict


def main():
    # Check adapter directory
    adapter_dir = Path("/home/user/ComfyUI/models/loras/omnix")
    if not adapter_dir.exists():
        adapter_dir = Path("C:/ComfyUI/models/loras/omnix")

    if not adapter_dir.exists():
        print(f"ERROR: Adapter directory not found: {adapter_dir}")
        print("Please update the path in this script.")
        return

    # Inspect each adapter
    adapters = list(adapter_dir.glob("*.safetensors"))
    if not adapters:
        print(f"ERROR: No .safetensors files found in {adapter_dir}")
        return

    print(f"\nFound {len(adapters)} adapter(s) in {adapter_dir}")

    for adapter_path in adapters:
        try:
            inspect_adapter_weights(adapter_path)
        except Exception as e:
            print(f"ERROR inspecting {adapter_path.name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
