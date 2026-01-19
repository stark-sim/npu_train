#!/usr/bin/env python3
"""
Download Qwen2.5 models from ModelScope with only essential formats
- safetensors (PyTorch weights)
- config, tokenizer, generation_config
"""

import os
import sys
from pathlib import Path

# Add modelscope to path if needed
sys.path.insert(0, '/home/sd/miniconda3/envs/npu_train/lib/python3.11/site-packages')

from modelscope.hub.snapshot_download import snapshot_download

# Model list - Qwen2.5 instruct versions
MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct", 
    "Qwen/Qwen2.5-14B-Instruct",
]

# Base directory for models
BASE_DIR = "/home/sd/npu_train/models"

# Essential file patterns to keep
ESSENTIAL_PATTERNS = [
    "*.safetensors",           # PyTorch weights
    "config.json",            # Model config
    "tokenizer.json",         # Tokenizer
    "tokenizer_config.json",  # Tokenizer config
    "vocab.json",             # Vocabulary
    "merges.txt",             # BPE merges
    "special_tokens_map.json", # Special tokens
    "generation_config.json", # Generation config
    "added_tokens.json",      # Added tokens
]

def download_model(model_name):
    """Download a single model with essential files only"""
    print(f"\n=== Downloading {model_name} ===")
    
    # Create model directory
    model_dir = Path(BASE_DIR) / model_name.replace("/", "-")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download with ModelScope
        snapshot_download(
            model_name,
            cache_dir=BASE_DIR,
            local_dir=model_dir,
        )
        
        # Clean up non-essential files
        print(f"Cleaning non-essential files for {model_name}...")
        clean_non_essential(model_dir)
        
        # Show final size
        size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / (1024**3)
        print(f"✅ {model_name} downloaded successfully! Size: {size:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return False

def clean_non_essential(model_dir):
    """Remove non-essential files to save space"""
    import fnmatch
    
    essential_files = set()
    for pattern in ESSENTIAL_PATTERNS:
        essential_files.update(model_dir.rglob(pattern))
    
    # Keep essential files and directories
    removed_count = 0
    removed_size = 0
    
    for file_path in model_dir.rglob('*'):
        if file_path.is_file() and file_path not in essential_files:
            # Check if it's an essential file by pattern matching
            is_essential = False
            for pattern in ESSENTIAL_PATTERNS:
                if fnmatch.fnmatch(file_path.name, pattern):
                    is_essential = True
                    break
            
            if not is_essential:
                file_size = file_path.stat().st_size
                file_path.unlink()
                removed_count += 1
                removed_size += file_size
    
    if removed_count > 0:
        print(f"   Removed {removed_count} non-essential files ({removed_size / (1024**3):.2f} GB)")

def main():
    """Main download function"""
    print("Starting Qwen2.5 model downloads...")
    print(f"Target directory: {BASE_DIR}")
    print(f"Models to download: {len(MODELS)}")
    
    # Check base directory
    base_path = Path(BASE_DIR)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Download models
    success_count = 0
    for model in MODELS:
        if download_model(model):
            success_count += 1
    
    # Summary
    print(f"\n=== Download Summary ===")
    print(f"Successfully downloaded: {success_count}/{len(MODELS)} models")
    
    # Total size
    total_size = 0
    for model_name in MODELS:
        model_dir = Path(BASE_DIR) / model_name.replace("/", "-")
        if model_dir.exists():
            total_size += sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
    
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    print(f"Models saved in: {BASE_DIR}")

if __name__ == "__main__":
    main()
