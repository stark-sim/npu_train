#!/usr/bin/env python3
"""
Simple resume download using ModelScope with better error handling
"""

import os
import sys
from pathlib import Path

# Add modelscope to path
sys.path.insert(0, '/home/sd/miniconda3/envs/npu_train/lib/python3.11/site-packages')

from modelscope.hub.snapshot_download import snapshot_download

BASE_DIR = "/home/sd/npu_train/models"

def download_model_with_retry(model_name, max_retries=3):
    """Download model with retry logic"""
    model_dir_name = model_name.replace("Qwen/Qwen2.5-", "Qwen-Qwen2.5-").replace("-Instruct", "-Instruct")
    model_dir = Path(BASE_DIR) / model_dir_name
    
    for attempt in range(max_retries):
        print(f"\n=== Attempt {attempt + 1}/{max_retries} for {model_name} ===")
        
        try:
            # Download with ModelScope
            snapshot_download(
                model_name,
                cache_dir=BASE_DIR,
                local_dir=model_dir,
            )
            
            # Check if we have safetensors files
            safetensors_files = list(model_dir.glob("*.safetensors"))
            if safetensors_files:
                total_size = sum(f.stat().st_size for f in safetensors_files)
                print(f"✅ Downloaded {len(safetensors_files)} safetensors files ({total_size / (1024**3):.2f} GB)")
                return True
            else:
                print(f"❌ No safetensors files found after download")
                if attempt < max_retries - 1:
                    print("Retrying...")
                    continue
                return False
                
        except Exception as e:
            print(f"❌ Download failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                continue
            return False
    
    return False

def main():
    """Download models with retry"""
    print("=== ModelScope Resume Download ===\n")
    
    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct"
    ]
    
    success_count = 0
    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Downloading: {model_name}")
        print(f"{'='*50}")
        
        if download_model_with_retry(model_name):
            success_count += 1
            print(f"✅ {model_name} completed successfully")
        else:
            print(f"❌ {model_name} failed after all retries")
    
    print(f"\n=== Final Summary ===")
    print(f"Successfully downloaded: {success_count}/{len(models)} models")
    
    # Show final status
    for model_name in models:
        model_dir_name = model_name.replace("Qwen/Qwen2.5-", "Qwen-Qwen2.5-").replace("-Instruct", "-Instruct")
        model_dir = Path(BASE_DIR) / model_dir_name
        
        if model_dir.exists():
            safetensors_files = list(model_dir.glob("*.safetensors"))
            if safetensors_files:
                total_size = sum(f.stat().st_size for f in safetensors_files)
                print(f"{model_dir_name}: {len(safetensors_files)} files, {total_size / (1024**3):.2f} GB")
            else:
                print(f"{model_dir_name}: No safetensors files")
        else:
            print(f"{model_dir_name}: Directory not found")

if __name__ == "__main__":
    main()
