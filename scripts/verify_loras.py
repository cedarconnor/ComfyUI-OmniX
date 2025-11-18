"""
Verify converted LoRA files have correct layer structure for ComfyUI FLUX
"""

import sys
from pathlib import Path
from safetensors.torch import load_file

def print_lora_info(file_path: Path, label: str):
    """Print information about a LoRA file"""
    print(f"\n{'='*70}")
    print(f"{label}: {file_path.name}")
    print(f"{'='*70}")

    try:
        state_dict = load_file(str(file_path))
    except Exception as e:
        print(f"[X] Failed to load: {e}")
        return None

    print(f"Total keys: {len(state_dict)}")
    print(f"Total parameters: {sum(p.numel() for p in state_dict.values()):,}")

    # Analyze key patterns
    key_patterns = {}
    for key in state_dict.keys():
        # Extract the pattern (first few components)
        parts = key.split('.')
        if len(parts) >= 2:
            pattern = '.'.join(parts[:2])
        else:
            pattern = parts[0]

        key_patterns[pattern] = key_patterns.get(pattern, 0) + 1

    print(f"\nKey patterns found:")
    for pattern, count in sorted(key_patterns.items()):
        print(f"  {pattern}: {count} keys")

    # Show first 10 keys as examples
    print(f"\nFirst 10 keys:")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        print(f"  {i+1}. {key}")

    # Check for ComfyUI FLUX patterns
    flux_patterns = ['lora_unet', 'lora_transformer', 'transformer', 'double_blocks', 'single_blocks']
    diffusers_patterns = ['lora_A', 'lora_B', 'alpha']

    has_flux = any(any(pattern in key for pattern in flux_patterns) for key in state_dict.keys())
    has_diffusers = any(any(pattern in key for pattern in diffusers_patterns) for key in state_dict.keys())

    print(f"\nFormat detection:")
    print(f"  Contains FLUX patterns: {has_flux}")
    print(f"  Contains Diffusers/PEFT patterns: {has_diffusers}")

    return state_dict

def compare_loras(original_path: Path, converted_path: Path):
    """Compare original and converted LoRA files"""
    print(f"\n{'#'*70}")
    print(f"COMPARING: {original_path.stem} -> {converted_path.stem}")
    print(f"{'#'*70}")

    original = print_lora_info(original_path, "ORIGINAL")
    converted = print_lora_info(converted_path, "CONVERTED")

    if original is None or converted is None:
        return

    # Compare key count
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"Original keys:  {len(original)}")
    print(f"Converted keys: {len(converted)}")
    print(f"Keys match: {len(original) == len(converted)}")

    # Check if any keys changed
    original_keys = set(original.keys())
    converted_keys = set(converted.keys())

    if original_keys == converted_keys:
        print("[OK] All keys are identical (no conversion applied)")
    else:
        added_keys = converted_keys - original_keys
        removed_keys = original_keys - converted_keys

        print(f"\nKeys removed: {len(removed_keys)}")
        if removed_keys and len(removed_keys) <= 10:
            for key in list(removed_keys)[:10]:
                print(f"  - {key}")

        print(f"\nKeys added: {len(added_keys)}")
        if added_keys and len(added_keys) <= 10:
            for key in list(added_keys)[:10]:
                print(f"  + {key}")

def main():
    # Files to verify
    tasks = [
        'text_to_pano',
        'rgb_to_depth',
        'rgb_to_normal',
        'rgb_to_albedo',
        'rgb_to_pbr',
        'rgb_to_semantic'
    ]

    omnix_dir = Path(r"C:\ComfyUI\models\loras\omnix")
    loras_dir = Path(r"C:\ComfyUI\models\loras")

    print("OmniX LoRA Verification Tool")
    print("="*70)

    # Check first task in detail
    task = tasks[0]  # text_to_pano
    original_file = omnix_dir / f"{task}_rgb.safetensors"
    converted_file = loras_dir / f"OmniX_{task}_comfyui.safetensors"

    if original_file.exists() and converted_file.exists():
        compare_loras(original_file, converted_file)

    # Quick check for all other tasks
    print(f"\n\n{'#'*70}")
    print("QUICK CHECK - ALL TASKS")
    print(f"{'#'*70}")

    for task in tasks:
        converted_file = loras_dir / f"OmniX_{task}_comfyui.safetensors"
        if converted_file.exists():
            try:
                state_dict = load_file(str(converted_file))
                keys_sample = list(state_dict.keys())[:3]
                print(f"\n{task}:")
                print(f"  File: {converted_file.name}")
                print(f"  Keys: {len(state_dict)}")
                print(f"  Sample keys:")
                for key in keys_sample:
                    print(f"    - {key}")
            except Exception as e:
                print(f"\n{task}: [X] Error - {e}")

if __name__ == "__main__":
    main()
