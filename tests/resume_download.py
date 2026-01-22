#!/usr/bin/env python3
"""
Resume download of Qwen2.5-7B model using ModelScope
This script will check and resume incomplete downloads automatically.
"""

from modelscope import snapshot_download
import os
import sys

# Model configuration
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
CACHE_DIR = "./models"

def main():
    print("=" * 60)
    print("ModelScope Resume Download for Qwen2.5-7B")
    print("=" * 60)
    print(f"Model ID: {MODEL_ID}")
    print(f"Cache Dir: {CACHE_DIR}")
    print("=" * 60)

    # ModelScope snapshot_download automatically handles resume
    # It will check existing files and only download missing parts
    try:
        print("\nStarting download (with resume support)...")
        print("This may take a while depending on your network speed.\n")

        model_dir = snapshot_download(
            MODEL_ID,
            cache_dir=CACHE_DIR,
            revision='master',  # Use master branch
        )

        print(f"\n{'=' * 60}")
        print(f"Download completed!")
        print(f"Model saved to: {model_dir}")
        print(f"{'=' * 60}")

        # Verify files
        print("\nVerifying downloaded files...")
        import json
        from safetensors import safe_open

        idx_file = os.path.join(model_dir, 'model.safetensors.index.json')
        with open(idx_file) as f:
            idx = json.load(f)

        expected_total = idx['metadata']['total_size']
        print(f"Expected total size: {expected_total / 1e9:.2f} GB")

        files_to_check = {}
        for key, filename in idx['weight_map'].items():
            if filename not in files_to_check:
                files_to_check[filename] = []
            files_to_check[filename].append(key)

        all_ok = True
        for filename in files_to_check.keys():
            filepath = os.path.join(model_dir, filename)
            actual_size = os.path.getsize(filepath)
            print(f"{filename}: {actual_size / 1e9:.2f} GB, {len(files_to_check[filename])} weights")

            # Try to verify the file
            try:
                with safe_open(filepath, framework='pt', device='cpu') as f:
                    keys = list(f.keys())
                    print(f"  OK: {len(keys)} weights loaded")
            except Exception as e:
                print(f"  ERROR: {e}")
                all_ok = False

        if all_ok:
            print("\n" + "=" * 60)
            print("All files verified successfully!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("WARNING: Some files are still corrupted!")
            print("Run this script again to continue downloading.")
            print("=" * 60)
            sys.exit(1)

    except Exception as e:
        print(f"\nError during download: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
