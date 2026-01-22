#!/usr/bin/env python3
"""
Download Qwen2.5-7B from ModelScope using wget
wget has better resume capability for slow networks
"""

import subprocess
import sys
import os
from pathlib import Path

# ModelScope CDN URL pattern
# https://modelscope.cn/api/v1/models/Qwen/Qwen2.5-7B-Instruct/repo?Path={filename}
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
BASE_URL = "https://modelscope.cn/api/v1/models/{}/repo".format(MODEL_ID)
CACHE_DIR = "./models/Qwen-Qwen2.5-7B-Instruct"

# Files to download
FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "LICENSE",
    "README.md",
    ".gitattributes",
    # Model weights (largest files)
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
]

# wget settings for slow networks
WGET_OPTS = [
    "--continue",           # Resume interrupted downloads
    "--tries=20",         # Retry 20 times
    "--timeout=300",       # 5 minute timeout
    "--read-timeout=600",  # 10 minute read timeout
    "--waitretry=10",      # Wait 10 seconds between retries
    "--retry-connrefused", # Retry on connection refused
    "--no-check-certificate",
    "--quiet",             # Less output
    "--show-progress",     # Show progress bar
    "-c",                # Continue mode (same as --continue)
]

def download_file(filename):
    """Download a single file using wget"""
    url = f"{BASE_URL}?Path={filename}"
    filepath = os.path.join(CACHE_DIR, filename)
    tmp_filepath = filepath + ".tmp"

    print(f"\n{'='*60}")
    print(f"Downloading: {filename}")
    print(f"{'='*60}")

    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Use wget to download
    cmd = ["wget"] + WGET_OPTS + ["-O", tmp_filepath, url]
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)

        # Rename to final filename
        if os.path.exists(tmp_filepath):
            os.rename(tmp_filepath, filepath)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"\nCompleted: {filename} ({size_mb:.1f} MB)")
            return True
        return False

    except subprocess.CalledProcessError as e:
        print(f"\nError downloading {filename}: {e}")
        return False

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("=" * 60)
    print("ModelScope Download for Qwen2.5-7B (using wget)")
    print("=" * 60)
    print(f"Cache Dir: {CACHE_DIR}")
    print(f"Files to download: {len(FILES)}")
    print("=" * 60)

    # Download small files first
    small_files = [f for f in FILES if not f.startswith("model-")]
    large_files = [f for f in FILES if f.startswith("model-")]

    print("\nPhase 1: Downloading small files...")
    for filename in small_files:
        download_file(filename)

    print("\nPhase 2: Downloading large model files...")
    print("These may take a long time on slow networks...")
    print("Press Ctrl+C to pause, script can be resumed later.\n")

    for filename in large_files:
        # Check if file already exists and is complete
        filepath = os.path.join(CACHE_DIR, filename)
        if os.path.exists(filepath) and not os.path.exists(filepath + ".tmp"):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"Skipping {filename} (already exists, {size_mb:.1f} MB)")
            continue

        success = download_file(filename)
        if not success:
            print(f"\nFailed to download {filename}")
            print(f"Re-run this script to resume from where it stopped.")
            return 1

    print("\n" + "=" * 60)
    print("All downloads completed!")
    print("=" * 60)

    # Verify total size
    total_size = sum(os.path.getsize(os.path.join(CACHE_DIR, f))
                     for f in FILES if os.path.exists(os.path.join(CACHE_DIR, f)))
    print(f"Total downloaded: {total_size / (1024**3):.2f} GB")

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        print("Run this script again to resume download.")
        sys.exit(1)
