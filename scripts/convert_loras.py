"""
LoRA Conversion Script for OmniX

Converts OmniX LoRAs from HuggingFace Diffusers/PEFT format to ComfyUI format.

Usage:
    python scripts/convert_loras.py --input /path/to/omnix/loras --output /path/to/comfyui/loras

Author: Cedar Connor
"""

import argparse
import os
from pathlib import Path
from safetensors.torch import load_file, save_file
import torch
from typing import Dict


def convert_lora_keys(state_dict: Dict[str, torch.Tensor], task_name: str) -> Dict[str, torch.Tensor]:
    """
    Convert LoRA keys from Diffusers format to ComfyUI FLUX format.

    Args:
        state_dict: Original state dict from Diffusers
        task_name: Name of the task (e.g., 'rgb_to_depth')

    Returns:
        Converted state dict for ComfyUI
    """
    converted = {}

    for key, value in state_dict.items():
        # OmniX LoRAs should already be in the correct format based on the
        # design doc note: "OmniX adapter keys already match Diffusers format"

        # However, we may need to handle some key remapping
        # ComfyUI FLUX LoRA keys typically look like:
        # - lora_unet_double_blocks_X_...
        # - lora_transformer_...

        new_key = key

        # Common conversions
        if 'lora_A' in key or 'lora_B' in key:
            # Already in PEFT format, should work
            new_key = key
        elif 'transformer' in key:
            # Transformer keys
            new_key = key
        elif 'unet' in key:
            # UNet keys (FLUX uses transformer, but some LoRAs may use unet naming)
            new_key = key.replace('unet', 'transformer')

        converted[new_key] = value

    return converted


def verify_lora(state_dict: Dict[str, torch.Tensor], task_name: str) -> bool:
    """
    Verify that the LoRA has valid keys and shapes.

    Args:
        state_dict: LoRA state dict
        task_name: Name of the task

    Returns:
        True if valid, False otherwise
    """
    if not state_dict:
        print(f"  [X] Empty state dict for {task_name}")
        return False

    # Check for common LoRA patterns
    has_lora_keys = any('lora' in key.lower() for key in state_dict.keys())
    if not has_lora_keys:
        print(f"  [!] Warning: No 'lora' keys found in {task_name}")

    # Check tensor shapes are reasonable
    for key, tensor in state_dict.items():
        if tensor.ndim not in [1, 2, 3, 4]:
            print(f"  [X] Invalid tensor shape for {key}: {tensor.shape}")
            return False

    print(f"  [OK] Found {len(state_dict)} parameters")
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"  [OK] Total parameters: {total_params:,}")

    return True


def convert_omnix_lora(input_path: Path, output_path: Path, task_name: str):
    """
    Convert a single OmniX LoRA file.

    Args:
        input_path: Path to input LoRA file
        output_path: Path to output LoRA file
        task_name: Name of the task
    """
    print(f"\nConverting {task_name}...")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")

    # Load the original LoRA
    try:
        state_dict = load_file(str(input_path))
    except Exception as e:
        print(f"  [X] Failed to load: {e}")
        return

    # Verify the LoRA
    if not verify_lora(state_dict, task_name):
        print(f"  [X] Verification failed for {task_name}")
        return

    # Convert keys
    converted = convert_lora_keys(state_dict, task_name)

    # Save converted LoRA
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(converted, str(output_path))
        print(f"  [OK] Saved successfully")
    except Exception as e:
        print(f"  [X] Failed to save: {e}")
        return


def main():
    parser = argparse.ArgumentParser(
        description="Convert OmniX LoRAs to ComfyUI format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to directory containing OmniX LoRA files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to ComfyUI loras directory"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=[
            "text_to_pano",
            "rgb_to_depth",
            "rgb_to_normal",
            "rgb_to_albedo",
            "rgb_to_pbr",
            "rgb_to_semantic"
        ],
        help="Tasks to convert (default: all)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    print("=" * 70)
    print("OmniX LoRA Conversion Tool")
    print("=" * 70)

    # Convert each task
    for task in args.tasks:
        # Try different possible file patterns
        possible_names = [
            f"{task}_rgb.safetensors",  # For tasks like text_to_pano_rgb
            f"lora_{task}.safetensors",
            f"OmniX_{task}.safetensors",
            f"{task}.safetensors",
            f"lora_{task}_rgb.safetensors",  # For tasks like lora_text_to_pano_rgb
        ]

        input_path = None
        for name in possible_names:
            candidate = input_dir / name
            if candidate.exists():
                input_path = candidate
                break

        # Also check subdirectories
        if not input_path:
            for subdir in input_dir.iterdir():
                if subdir.is_dir() and task in subdir.name:
                    # Look for adapter_model.safetensors in subdirectory
                    adapter = subdir / "adapter_model.safetensors"
                    if adapter.exists():
                        input_path = adapter
                        break

        if not input_path:
            print(f"\n[!] Skipping {task}: LoRA file not found")
            print(f"  Searched for: {', '.join(possible_names)}")
            continue

        # Determine output filename
        output_filename = f"OmniX_{task}_comfyui.safetensors"
        output_path = output_dir / output_filename

        # Convert
        convert_omnix_lora(input_path, output_path, task)

    print("\n" + "=" * 70)
    print("Conversion complete!")
    print("=" * 70)
    print(f"\nConverted LoRAs saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Copy the converted LoRAs to your ComfyUI/models/loras/ directory")
    print("2. Restart ComfyUI")
    print("3. Use 'Load LoRA' nodes in your workflows")


if __name__ == "__main__":
    main()
