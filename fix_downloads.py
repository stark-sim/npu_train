#!/usr/bin/env python3
"""
Fix incomplete downloads by checking file integrity and re-downloading corrupted files
"""

import os
import sys
import hashlib
from pathlib import Path

# Add modelscope to path
sys.path.insert(0, '/home/sd/miniconda3/envs/npu_train/lib/python3.11/site-packages')

from modelscope.hub.snapshot_download import snapshot_download

BASE_DIR = "/home/sd/npu_train/models"

def get_file_hash(filepath, chunk_size=8192):
    """Calculate SHA256 hash of file"""
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {filepath}: {e}")
        return None

def check_safetensors_integrity(model_dir):
    """Check if safetensors files are complete"""
    model_path = Path(model_dir)
    corrupted_files = []
    
    for safetensor_file in model_path.glob("*.safetensors"):
        print(f"Checking {safetensor_file.name}...")
        
        # Try to read the file header to check integrity
        try:
            with open(safetensor_file, 'rb') as f:
                # Read first few bytes to check if it's a valid safetensors file
                header = f.read(8)
                if len(header) < 8:
                    print(f"  ❌ File too small or incomplete")
                    corrupted_files.append(safetensor_file)
                    continue
                
                # Check if it starts with the expected safetensors magic
                if not header.startswith(b'{' if header[0] < 128 else b'\x00'):
                    print(f"  ❌ Invalid safetensors header")
                    corrupted_files.append(safetensor_file)
                    continue
                
                # Try to read the full header length
                f.seek(0)
                header_len_bytes = f.read(8)
                if len(header_len_bytes) < 8:
                    print(f"  ❌ Cannot read header length")
                    corrupted_files.append(safetensor_file)
                    continue
                
                # Read the header
                import struct
                header_len = struct.unpack('<Q', header_len_bytes)[0]
                if header_len > 100 * 1024 * 1024:  # 100MB limit for header
                    print(f"  ❌ Header too large ({header_len} bytes)")
                    corrupted_files.append(safetensor_file)
                    continue
                
                header_data = f.read(header_len)
                if len(header_data) < header_len:
                    print(f"  ❌ Incomplete header ({len(header_data)}/{header_len} bytes)")
                    corrupted_files.append(safetensor_file)
                    continue
                
                print(f"  ✅ File appears complete")
                
        except Exception as e:
            print(f"  ❌ Error checking file: {e}")
            corrupted_files.append(safetensor_file)
    
    return corrupted_files

def fix_model(model_name):
    """Fix a specific model by re-downloading corrupted files"""
    print(f"\n=== Fixing {model_name} ===")
    
    model_dir = Path(BASE_DIR) / model_name
    
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        return False
    
    # Check integrity
    corrupted_files = check_safetensors_integrity(model_dir)
    
    if not corrupted_files:
        print(f"✅ All files in {model_name} appear complete")
        return True
    
    print(f"Found {len(corrupted_files)} corrupted files:")
    for f in corrupted_files:
        print(f"  - {f.name}")
    
    # Remove corrupted files
    for f in corrupted_files:
        try:
            f.unlink()
            print(f"Removed corrupted file: {f.name}")
        except Exception as e:
            print(f"Failed to remove {f.name}: {e}")
    
    # Re-download the model (ModelScope should resume)
    print(f"Re-downloading {model_name}...")
    try:
        snapshot_download(
            model_name.replace("Qwen-Qwen2.5-", "Qwen/Qwen2.5-").replace("-Instruct", "-Instruct"),
            cache_dir=BASE_DIR,
            local_dir=model_dir,
        )
        
        # Verify again
        new_corrupted = check_safetensors_integrity(model_dir)
        if not new_corrupted:
            print(f"✅ {model_name} fixed successfully!")
            return True
        else:
            print(f"❌ {model_name} still has {len(new_corrupted)} corrupted files")
            return False
            
    except Exception as e:
        print(f"❌ Failed to re-download {model_name}: {e}")
        return False

def main():
    """Fix all Qwen models"""
    print("=== Fixing Qwen Model Downloads ===\n")
    
    models = [
        "Qwen-Qwen2.5-7B-Instruct",
        "Qwen-Qwen2.5-14B-Instruct"
    ]
    
    success_count = 0
    for model_name in models:
        if fix_model(model_name):
            success_count += 1
    
    print(f"\n=== Fix Summary ===")
    print(f"Successfully fixed: {success_count}/{len(models)} models")
    
    # Show final sizes
    total_size = 0
    for model_name in models:
        model_dir = Path(BASE_DIR) / model_name
        if model_dir.exists():
            size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
            total_size += size
            print(f"{model_name}: {size / (1024**3):.2f} GB")
    
    print(f"Total size: {total_size / (1024**3):.2f} GB")

if __name__ == "__main__":
    main()
