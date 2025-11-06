#!/usr/bin/env python3
"""
OmniX Model Downloader for ComfyUI

Downloads OmniX adapter weights from HuggingFace and organizes them
for use with ComfyUI-OmniX custom nodes.

Usage:
    python download_models.py
    python download_models.py --output_dir /path/to/ComfyUI/models/omnix
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
import shutil


# HuggingFace repository
REPO_ID = "KevinHuang/OmniX"

# Expected adapter files (based on OmniX architecture)
# This will be updated based on actual repo structure
ADAPTER_FILES = [
    "rgb_adapter.safetensors",
    "distance_adapter.safetensors",
    "normal_adapter.safetensors",
    "albedo_adapter.safetensors",
    "roughness_adapter.safetensors",
    "metallic_adapter.safetensors",
    "semantic_adapter.safetensors",
    "config.json",
]

# File renaming map (if needed)
# Format: {"original_name": "target_name"}
FILE_RENAME_MAP = {
    # Will be populated based on actual file names
    # Example: "rgb.safetensors": "rgb_adapter.safetensors"
}


def list_available_files(repo_id: str):
    """List all files in the HuggingFace repository"""
    print(f"Listing files in {repo_id}...")
    try:
        files = list_repo_files(repo_id)
        print(f"\nFound {len(files)} files:")
        for f in sorted(files):
            print(f"  - {f}")
        return files
    except Exception as e:
        print(f"Error listing files: {e}")
        return []


def download_file(repo_id: str, filename: str, output_dir: Path):
    """Download a single file from HuggingFace"""
    try:
        print(f"Downloading {filename}...")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=None,  # Use default cache
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        print(f"  ✓ Downloaded to {local_path}")
        return local_path
    except Exception as e:
        print(f"  ✗ Failed to download {filename}: {e}")
        return None


def rename_file(src_path: Path, new_name: str):
    """Rename a file to match expected naming convention"""
    if src_path.name == new_name:
        return src_path

    dst_path = src_path.parent / new_name
    print(f"Renaming {src_path.name} -> {new_name}")
    shutil.move(str(src_path), str(dst_path))
    return dst_path


def main():
    parser = argparse.ArgumentParser(
        description="Download OmniX adapter weights for ComfyUI"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: ./models/omnix/omnix-base)"
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list available files without downloading"
    )
    parser.add_argument(
        "--adapters",
        nargs="+",
        default=None,
        help="Specific adapters to download (default: all)"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Try to find ComfyUI directory
        possible_paths = [
            Path.home() / "ComfyUI" / "models" / "omnix" / "omnix-base",
            Path("models") / "omnix" / "omnix-base",
            Path("ComfyUI") / "models" / "omnix" / "omnix-base",
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
    print(f"Output directory: {output_dir}")
    print()

    # List available files
    available_files = list_available_files(REPO_ID)

    if args.list_only:
        print("\nUse this information to update download_models.py if needed.")
        return

    # Determine which files to download
    if args.adapters:
        files_to_download = []
        for adapter in args.adapters:
            # Find matching file
            matching = [f for f in available_files if adapter in f.lower()]
            if matching:
                files_to_download.extend(matching)
            else:
                print(f"Warning: No file found for adapter '{adapter}'")
    else:
        # Download all relevant files
        # Filter for .safetensors and config files
        files_to_download = [
            f for f in available_files
            if f.endswith('.safetensors') or f.endswith('.json')
        ]

    if not files_to_download:
        print("No files to download!")
        print("Use --list-only to see available files.")
        return

    print(f"\nDownloading {len(files_to_download)} files...\n")

    # Download files
    downloaded_files = []
    for filename in files_to_download:
        local_path = download_file(REPO_ID, filename, output_dir)
        if local_path:
            downloaded_files.append(Path(local_path))

    # Rename files if needed
    print("\nApplying file naming conventions...")
    for file_path in downloaded_files:
        if file_path.name in FILE_RENAME_MAP:
            new_name = FILE_RENAME_MAP[file_path.name]
            rename_file(file_path, new_name)

    print("\n" + "="*60)
    print("Download complete!")
    print(f"Files saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Ensure files are in: ComfyUI/models/omnix/omnix-base/")
    print("2. Restart ComfyUI if it's running")
    print("3. Use OmniXAdapterLoader node in your workflows")
    print("="*60)


if __name__ == "__main__":
    main()
