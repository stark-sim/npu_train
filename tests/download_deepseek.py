#!/usr/bin/env python3
"""
Download DeepSeek-Coder-V2-Lite-Instruct model via ModelScope
"""

import os
from pathlib import Path

# Disable torch_npu auto-load for download
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

try:
    from modelscope import snapshot_download
except ImportError:
    print("Installing modelscope...")
    os.system("pip install modelscope -q")
    from modelscope import snapshot_download

MODEL_ID = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
MODEL_PATH = "/home/sd/npu_train/models/DeepSeek-Coder-V2-Lite-Instruct"

print("=" * 60)
print("Downloading DeepSeek-Coder-V2-Lite-Instruct")
print("=" * 60)
print(f"Model ID: {MODEL_ID}")
print(f"Target path: {MODEL_PATH}")
print("=" * 60)

model_path = Path(MODEL_PATH)
model_path.mkdir(parents=True, exist_ok=True)

try:
    snapshot_download(
        MODEL_ID,
        local_dir=MODEL_PATH,
    )

    print("\nDownload completed!")
    print(f"Model saved to: {MODEL_PATH}")

    # Show downloaded files
    print("\nDownloaded files:")
    for f in sorted(model_path.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024*1024)
            print(f"  {f.relative_to(model_path)}: {size_mb:.1f} MB")

except Exception as e:
    print(f"\nDownload failed: {e}")
    print("You may need to retry or check network connection")
