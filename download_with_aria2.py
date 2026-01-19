#!/usr/bin/env python3
"""
Download models using aria2c for better resume capability and parallel downloads
"""

import os
import sys
import json
import subprocess
from pathlib import Path

BASE_DIR = "/home/sd/npu_train/models"

def get_model_files(model_name):
    """Get list of files to download for a model"""
    # Use ModelScope API to get file list
    try:
        import requests
        
        # Get file list from ModelScope API
        api_url = f"https://www.modelscope.cn/api/v1/models/{model_name}/repo/files"
        response = requests.get(api_url, timeout=30)
        
        if response.status_code == 200:
            files_data = response.json()
            files = []
            
            for file_info in files_data.get('Data', {}).get('Files', []):
                if file_info.get('Type') == 'file':
                    file_path = file_info.get('Path', '')
                    # Only download essential files
                    if (file_path.endswith('.safetensors') or 
                        file_path.endswith('.json') or 
                        file_path.endswith('.txt')):
                        files.append({
                            'path': file_path,
                            'size': file_info.get('Size', 0)
                        })
            
            return files
        else:
            print(f"Failed to get file list for {model_name}: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"Error getting file list for {model_name}: {e}")
        return []

def download_with_aria2(model_name, files):
    """Download files using aria2c"""
    model_dir = Path(BASE_DIR) / model_name.replace("Qwen/Qwen2.5-", "Qwen-Qwen2.5-").replace("-Instruct", "-Instruct")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create aria2 input file
    aria_input = model_dir / "download.txt"
    
    with open(aria_input, 'w') as f:
        for file_info in files:
            file_path = file_info['path']
            # ModelScope download URL
            download_url = f"https://www.modelscope.cn/{model_name}/resolve/master/{file_path}"
            local_path = model_dir / file_path
            
            # Create parent directory if needed
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            f.write(f"{download_url}\n")
            f.write(f"  dir={model_dir}\n")
            f.write(f"  out={file_path}\n")
            f.write(f"  continue=true\n")
            f.write(f"  max-tries=5\n")
            f.write(f"  retry-wait=10\n")
            f.write(f"  timeout=600\n")
            f.write(f"  split=16\n")
            f.write(f"  max-connection-per-server=16\n")
            f.write("\n")
    
    # Run aria2c
    cmd = [
        "aria2c",
        "-i", str(aria_input),
        "-j", "8",  # Max concurrent downloads
        "--file-allocation=none",
        "--max-tries=5",
        "--retry-wait=10",
        "--timeout=600",
        "--continue=true",
        "--check-integrity=true",
        "--console-log-level=notice"
    ]
    
    print(f"Downloading {model_name} with aria2c...")
    try:
        result = subprocess.run(cmd, cwd=model_dir, capture_output=True, text=True, timeout=7200)
        
        if result.returncode == 0:
            print(f"✅ {model_name} downloaded successfully")
            return True
        else:
            print(f"❌ aria2c failed for {model_name}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ aria2c timeout for {model_name}")
        return False
    except Exception as e:
        print(f"❌ aria2c error for {model_name}: {e}")
        return False

def main():
    """Download models using aria2c"""
    print("=== Downloading Models with aria2c ===\n")
    
    # Check if aria2c is available
    try:
        subprocess.run(["aria2c", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ aria2c not found. Please install it:")
        print("   sudo apt-get install aria2  # Ubuntu/Debian")
        print("   sudo yum install aria2      # CentOS/RHEL")
        return
    
    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct"
    ]
    
    success_count = 0
    for model_name in models:
        print(f"\n=== Processing {model_name} ===")
        
        # Get file list
        files = get_model_files(model_name)
        
        if not files:
            print(f"❌ No files found for {model_name}")
            continue
        
        print(f"Found {len(files)} files to download")
        
        # Download with aria2c
        if download_with_aria2(model_name, files):
            success_count += 1
    
    print(f"\n=== Summary ===")
    print(f"Successfully downloaded: {success_count}/{len(models)} models")

if __name__ == "__main__":
    main()
