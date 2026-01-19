#!/usr/bin/env python3
"""
Download models using wget with resume capability
"""

import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = "/home/sd/npu_train/models"

def download_file_with_wget(url, local_path, max_retries=3):
    """Download a single file using wget with resume capability"""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "wget",
        "-c",  # Continue/resume
        "-t", str(max_retries),  # Retries
        "--timeout=300",  # 5 minutes timeout
        "--progress=bar:force",
        "--no-check-certificate",
        "-O", str(local_path),
        url
    ]
    
    print(f"Downloading: {url}")
    print(f"Saving to: {local_path}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print(f"✅ Downloaded: {local_path.name}")
            return True
        else:
            print(f"❌ Failed to download: {url}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout downloading: {url}")
        return False
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False

def download_model_files(model_name, files):
    """Download all files for a model"""
    model_dir = Path(BASE_DIR) / model_name.replace("Qwen/Qwen2.5-", "Qwen-Qwen2.5-").replace("-Instruct", "-Instruct")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    total_files = len(files)
    
    for i, file_info in enumerate(files, 1):
        file_path = file_info['path']
        download_url = f"https://www.modelscope.cn/{model_name}/resolve/master/{file_path}"
        local_path = model_dir / file_path
        
        print(f"\n[{i}/{total_files}] Processing {file_path}")
        
        if download_file_with_wget(download_url, local_path):
            success_count += 1
        else:
            print(f"Failed to download {file_path}")
    
    print(f"\nCompleted: {success_count}/{total_files} files downloaded")
    return success_count == total_files

def get_essential_files(model_name):
    """Get essential files for a model (hardcoded for reliability)"""
    if "7B" in model_name:
        return [
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors", 
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json"
        ]
    elif "14B" in model_name:
        return [
            "model-00001-of-00008.safetensors",
            "model-00002-of-00008.safetensors",
            "model-00003-of-00008.safetensors", 
            "model-00004-of-00008.safetensors",
            "model-00005-of-00008.safetensors",
            "model-00006-of-00008.safetensors",
            "model-00007-of-00008.safetensors",
            "model-00008-of-00008.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json"
        ]
    else:
        return []

def main():
    """Download models using wget"""
    print("=== Downloading Models with wget ===\n")
    
    # Check if wget is available
    try:
        subprocess.run(["wget", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ wget not found")
        return
    
    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct"
    ]
    
    success_count = 0
    for model_name in models:
        print(f"\n=== Processing {model_name} ===")
        
        # Get essential files
        files = [{"path": f} for f in get_essential_files(model_name)]
        
        if not files:
            print(f"❌ No files defined for {model_name}")
            continue
        
        print(f"Will download {len(files)} essential files")
        
        # Download files
        if download_model_files(model_name, files):
            success_count += 1
            print(f"✅ {model_name} completed successfully")
        else:
            print(f"❌ {model_name} had some failures")
    
    print(f"\n=== Summary ===")
    print(f"Successfully downloaded: {success_count}/{len(models)} models")

if __name__ == "__main__":
    main()
