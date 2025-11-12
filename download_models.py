#!/usr/bin/env python3
"""
OmniX Model Downloader for ComfyUI

Downloads OmniX adapter weights from HuggingFace and organizes them
for use with ComfyUI-OmniX custom nodes.

Repository: https://huggingface.co/KevinHuang/OmniX

Usage:
    python download_models.py
    python download_models.py --output_dir /path/to/ComfyUI/models/loras/omnix
    python download_models.py --list-only
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

# Files to download from HuggingFace (keeping original names)
ADAPTER_FILES = [
    "text_to_pano_rgb.safetensors",
    "rgb_to_depth_depth.safetensors",
    "rgb_to_normal_normal.safetensors",
    "rgb_to_albedo_albedo.safetensors",
    "rgb_to_pbr_pbr.safetensors",
    "rgb_to_semantic_semantic.safetensors",
]

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


def download_adapters(repo_id: str, output_dir: Path, files_to_download=None):
    """
    Download adapter files from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID
        output_dir: Local output directory
        files_to_download: List of specific files to download (optional)
    """
    if files_to_download is None:
        files_to_download = ADAPTER_FILES

    print(f"\n📦 Downloading {len(files_to_download)} adapter file(s)...")

    try:
        # Try to check if repository exists first
        from huggingface_hub import repo_exists

        if not repo_exists(repo_id=repo_id, repo_type="model"):
            print(f"  ⚠️  Repository {repo_id} not found on HuggingFace")
            print(f"  ℹ️  The official OmniX weights may not be publicly released yet.")
            print(f"  ℹ️  Check https://github.com/HKU-MMLab/OmniX for updates")
            return False

        # Download all files directly to output directory
        print(f"  Downloading to: {output_dir}")

        snapshot_download(
            repo_id=repo_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            allow_patterns=["*.safetensors"] if files_to_download == ADAPTER_FILES else files_to_download
        )

        # Verify downloaded files
        downloaded_files = []
        for filename in files_to_download:
            file_path = output_dir / filename
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"    ✓ {filename} ({size_mb:.1f} MB)")
                downloaded_files.append(filename)
            else:
                print(f"    ⚠️  {filename} not found")

        if downloaded_files:
            print(f"  ✅ Downloaded {len(downloaded_files)}/{len(files_to_download)} files successfully")
            return True
        else:
            print(f"  ❌ No files downloaded")
            return False

    except Exception as e:
        error_msg = str(e)
        print(f"  ❌ Failed to download: {error_msg}")

        # Provide helpful error messages
        if "Repository Not Found" in error_msg or "404" in error_msg:
            print(f"\n  ℹ️  Troubleshooting:")
            print(f"     - OmniX weights may not be publicly released yet")
            print(f"     - Check https://arxiv.org/abs/2510.26800 for paper")
            print(f"     - Check https://github.com/HKU-MMLab/OmniX for official release")
            print(f"     - If you have access to weights, place them manually in: {output_dir}")
        elif "Unauthorized" in error_msg or "403" in error_msg:
            print(f"\n  ℹ️  Repository requires authentication:")
            print(f"     Run: huggingface-cli login")
        elif "Connection" in error_msg or "timeout" in error_msg.lower():
            print(f"\n  ℹ️  Network error:")
            print(f"     - Check internet connection")
            print(f"     - Try again later")
            print(f"     - Use VPN if HuggingFace is blocked")

        return False


def download_all_adapters(repo_id: str, output_dir: Path):
    """
    Download all adapter files.

    Args:
        repo_id: HuggingFace repository ID
        output_dir: Local output directory
    """
    print(f"\n{'='*60}")
    print(f"Downloading OmniX adapters from {repo_id}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

    # Download all adapter files
    success = download_adapters(repo_id, output_dir)

    if success:
        print(f"\n{'='*60}")
        print(f"✅ Download complete!")
        print(f"   Location: {output_dir}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"❌ Download failed!")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Download OmniX adapter weights for ComfyUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all adapters to default location
  python download_models.py

  # Download to custom location
  python download_models.py --output_dir /path/to/ComfyUI/models/loras/omnix

  # List available files
  python download_models.py --list-only

Files downloaded (keeping original HuggingFace names):
  text_to_pano_rgb.safetensors           - Text/image to panorama generation
  rgb_to_depth_depth.safetensors         - Depth/distance prediction
  rgb_to_normal_normal.safetensors       - Normal map estimation
  rgb_to_albedo_albedo.safetensors       - Albedo/base color extraction
  rgb_to_pbr_pbr.safetensors             - PBR material properties
  rgb_to_semantic_semantic.safetensors   - Semantic segmentation
"""
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: ./models/loras/omnix)"
    )

    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List available files and exit"
    )

    args = parser.parse_args()

    # List files
    if args.list_only:
        print("\n📋 OmniX Adapter Files (keeping HuggingFace names):\n")
        print(f"{'Filename':<40} {'Purpose'}")
        print("-" * 80)
        for filename in ADAPTER_FILES:
            purpose = {
                "text_to_pano_rgb.safetensors": "Panorama generation",
                "rgb_to_depth_depth.safetensors": "Depth prediction",
                "rgb_to_normal_normal.safetensors": "Normal estimation",
                "rgb_to_albedo_albedo.safetensors": "Albedo extraction",
                "rgb_to_pbr_pbr.safetensors": "Roughness & metallic (PBR)",
                "rgb_to_semantic_semantic.safetensors": "Semantic segmentation",
            }.get(filename, "Unknown")
            print(f"{filename:<40} {purpose}")
        print(f"\nTotal: {len(ADAPTER_FILES)} files (~{len(ADAPTER_FILES) * 224} MB)\n")
        return

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Try to find ComfyUI directory
        possible_paths = [
            Path.home() / "ComfyUI" / "models" / "loras" / "omnix",
            Path("ComfyUI") / "models" / "loras" / "omnix",
            Path("models") / "loras" / "omnix",
        ]

        output_dir = None
        for path in possible_paths:
            if path.parent.parent.exists():  # Check if models/ exists
                output_dir = path
                break

        if output_dir is None:
            output_dir = Path("models") / "loras" / "omnix"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download adapters
    try:
        download_all_adapters(REPO_ID, output_dir)

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
