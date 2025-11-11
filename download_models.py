#!/usr/bin/env python3
"""
OmniX Model Downloader for ComfyUI

Downloads OmniX adapter weights from HuggingFace and organizes them
for use with ComfyUI-OmniX custom nodes.

Repository: https://huggingface.co/KevinHuang/OmniX

Usage:
    python download_models.py
    python download_models.py --output_dir /path/to/ComfyUI/models/omnix/omnix-base
    python download_models.py --adapters rgb distance normal
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
import shutil


# HuggingFace repository
# NOTE: This is a placeholder. The actual OmniX repository may be different.
# Check https://github.com/HKU-MMLab/OmniX for official release information.
REPO_ID = "KevinHuang/OmniX"  # Update with actual repository when available

# Alternative: Official OmniX repository (update when released)
OFFICIAL_REPO = "HKU-MMLab/OmniX"

# Adapter directory mapping (HuggingFace dir -> our adapter name)
ADAPTER_DIR_MAP = {
    "image_to_pano": "rgb_generation",
    "rgb_to_depth": "distance",
    "rgb_to_normal": "normal",
    "rgb_to_albedo": "albedo",
    "rgb_to_pbr": "pbr",  # Contains both roughness and metallic
    "rgb_to_semantic": "semantic",
}

# Alternative mapping for user-friendly names
USER_FRIENDLY_NAMES = {
    "rgb": "image_to_pano",
    "rgb_generation": "image_to_pano",
    "generation": "image_to_pano",
    "depth": "rgb_to_depth",
    "distance": "rgb_to_depth",
    "normal": "rgb_to_normal",
    "albedo": "rgb_to_albedo",
    "pbr": "rgb_to_pbr",
    "roughness": "rgb_to_pbr",
    "metallic": "rgb_to_pbr",
    "semantic": "rgb_to_semantic",
    "segmentation": "rgb_to_semantic",
}


def download_adapter_directory(repo_id: str, adapter_dir: str, output_dir: Path):
    """
    Download a specific adapter directory from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID
        adapter_dir: Directory name in the repo (e.g., "image_to_pano")
        output_dir: Local output directory
    """
    print(f"\n📦 Downloading {adapter_dir}...")

    try:
        # Try to check if repository exists first
        from huggingface_hub import repo_exists

        if not repo_exists(repo_id=repo_id, repo_type="model"):
            print(f"  ⚠️  Repository {repo_id} not found on HuggingFace")
            print(f"  ℹ️  The official OmniX weights may not be publicly released yet.")
            print(f"  ℹ️  Check https://github.com/HKU-MMLab/OmniX for updates")
            return False

        # Download the specific directory
        local_path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=f"{adapter_dir}/**",
            local_dir=output_dir / adapter_dir,
            local_dir_use_symlinks=False
        )

        # Move files from subdirectory to output_dir with proper naming
        adapter_name = ADAPTER_DIR_MAP.get(adapter_dir, adapter_dir)
        source_dir = output_dir / adapter_dir / adapter_dir

        if source_dir.exists():
            # List downloaded files
            files = list(source_dir.glob("*"))
            print(f"  Downloaded {len(files)} files:")

            for file_path in files:
                if file_path.is_file():
                    # Rename to our convention: {adapter_name}_adapter.safetensors
                    if file_path.suffix == ".safetensors":
                        new_name = f"{adapter_name}_adapter.safetensors"
                        target_path = output_dir / new_name
                        shutil.copy2(file_path, target_path)
                        print(f"    ✓ {file_path.name} → {new_name}")
                    elif file_path.name == "config.json":
                        # Keep config files with adapter prefix
                        target_path = output_dir / f"{adapter_name}_config.json"
                        shutil.copy2(file_path, target_path)
                        print(f"    ✓ {file_path.name} → {target_path.name}")
                    else:
                        # Copy other files as-is
                        target_path = output_dir / file_path.name
                        shutil.copy2(file_path, target_path)
                        print(f"    ✓ {file_path.name}")

            # Clean up the temporary directory
            shutil.rmtree(output_dir / adapter_dir)

        print(f"  ✅ {adapter_dir} downloaded successfully")
        return True

    except Exception as e:
        error_msg = str(e)
        print(f"  ❌ Failed to download {adapter_dir}: {error_msg}")

        # Provide helpful error messages
        if "Repository Not Found" in error_msg or "404" in error_msg:
            print(f"\n  ℹ️  Troubleshooting:")
            print(f"     - OmniX weights may not be publicly released yet")
            print(f"     - Check https://arxiv.org/abs/2510.26800 for paper")
            print(f"     - Check https://github.com/HKU-MMLab/OmniX for official release")
            print(f"     - If you have access to weights, place them manually in:")
            print(f"       {output_dir}/{adapter_name}_adapter.safetensors")
        elif "Unauthorized" in error_msg or "403" in error_msg:
            print(f"\n  ℹ️  Repository requires authentication:")
            print(f"     Run: huggingface-cli login")
        elif "Connection" in error_msg or "timeout" in error_msg.lower():
            print(f"\n  ℹ️  Network error:")
            print(f"     - Check internet connection")
            print(f"     - Try again later")
            print(f"     - Use VPN if HuggingFace is blocked")

        return False


def download_all_adapters(repo_id: str, output_dir: Path, specific_adapters=None):
    """
    Download all or specific adapters.

    Args:
        repo_id: HuggingFace repository ID
        output_dir: Local output directory
        specific_adapters: List of specific adapters to download (optional)
    """
    # Determine which adapters to download
    if specific_adapters:
        # Map user-friendly names to actual directory names
        adapter_dirs = []
        for adapter in specific_adapters:
            adapter_lower = adapter.lower()
            if adapter_lower in USER_FRIENDLY_NAMES:
                dir_name = USER_FRIENDLY_NAMES[adapter_lower]
                if dir_name not in adapter_dirs:
                    adapter_dirs.append(dir_name)
            elif adapter in ADAPTER_DIR_MAP:
                adapter_dirs.append(adapter)
            else:
                print(f"⚠️  Warning: Unknown adapter '{adapter}', skipping")
    else:
        # Download all adapters
        adapter_dirs = list(ADAPTER_DIR_MAP.keys())

    if not adapter_dirs:
        print("❌ No valid adapters specified!")
        return

    print(f"\n{'='*60}")
    print(f"Downloading {len(adapter_dirs)} adapter(s) from {repo_id}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

    # Download each adapter
    success_count = 0
    for adapter_dir in adapter_dirs:
        if download_adapter_directory(repo_id, adapter_dir, output_dir):
            success_count += 1

    print(f"\n{'='*60}")
    print(f"✅ Download complete!")
    print(f"   Successfully downloaded: {success_count}/{len(adapter_dirs)} adapters")
    print(f"   Location: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Download OmniX adapter weights for ComfyUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all adapters
  python download_models.py

  # Download specific adapters
  python download_models.py --adapters rgb depth normal albedo

  # Download to custom location
  python download_models.py --output_dir /path/to/ComfyUI/models/omnix/omnix-base

Available adapters:
  rgb, generation          - Text/image to panorama generation
  depth, distance          - Depth/distance prediction
  normal                   - Normal map estimation
  albedo                   - Albedo/base color extraction
  pbr, roughness, metallic - PBR material properties
  semantic, segmentation   - Semantic segmentation
"""
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: ./models/omnix/omnix-base)"
    )

    parser.add_argument(
        "--adapters",
        nargs="+",
        default=None,
        help="Specific adapters to download (default: all)"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available adapters and exit"
    )

    args = parser.parse_args()

    # List adapters
    if args.list:
        print("\n📋 Available Adapters:\n")
        print(f"{'Adapter Name':<20} {'HF Directory':<25} {'Purpose'}")
        print("-" * 70)
        print(f"{'rgb_generation':<20} {'image_to_pano':<25} Panorama generation")
        print(f"{'distance':<20} {'rgb_to_depth':<25} Depth prediction")
        print(f"{'normal':<20} {'rgb_to_normal':<25} Normal estimation")
        print(f"{'albedo':<20} {'rgb_to_albedo':<25} Albedo extraction")
        print(f"{'pbr':<20} {'rgb_to_pbr':<25} Roughness & metallic")
        print(f"{'semantic':<20} {'rgb_to_semantic':<25} Segmentation")
        print("\nAliases: rgb/generation, depth/distance, pbr/roughness/metallic, semantic/segmentation\n")
        return

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Try to find ComfyUI directory
        possible_paths = [
            Path.home() / "ComfyUI" / "models" / "omnix" / "omnix-base",
            Path("ComfyUI") / "models" / "omnix" / "omnix-base",
            Path("models") / "omnix" / "omnix-base",
        ]

        output_dir = None
        for path in possible_paths:
            if path.parent.parent.exists():  # Check if models/ exists
                output_dir = path
                break

        if output_dir is None:
            output_dir = Path("models") / "omnix" / "omnix-base"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download adapters
    try:
        download_all_adapters(REPO_ID, output_dir, args.adapters)

        print("\n📝 Next steps:")
        print("1. Verify files in:", output_dir)
        print("2. Restart ComfyUI if it's running")
        print("3. Add 'OmniXAdapterLoader' node to your workflow")
        print("4. Select 'omnix-base' preset and 'fp16' precision")
        print("\n🎉 You're ready to generate panoramas!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Install huggingface_hub: pip install huggingface_hub")
        print("2. If repo is gated, login: huggingface-cli login")
        print("3. Check network connection")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
