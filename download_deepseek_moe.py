#!/usr/bin/env python3
"""
Quick script to download DeepSeek-V2-Lite for MoE TP training

DeepSeek-V2-Lite specs:
- Total parameters: 16B
- Activated per token: 2B (MoE with 64 experts, top-6)
- Context length: 32K
"""

import os
import sys
from pathlib import Path

# Add modelscope to path
sys.path.insert(0, '/home/sd/miniconda3/envs/npu_train/lib/python3.11/site-packages')

from modelscope.hub.snapshot_download import snapshot_download

# Model configuration
MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite"
BASE_DIR = "/home/sd/npu_train/models"
MODEL_DIR = Path(BASE_DIR) / "deepseek-ai-DeepSeek-V2-Lite"

# Essential files for MoE models
ESSENTIAL_PATTERNS = [
    "*.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "generation_config.json",
    "added_tokens.json",
]

def download_deepseek_v2_lite():
    """Download DeepSeek-V2-Lite model"""
    print("=" * 60)
    print("DeepSeek-V2-Lite Download Script")
    print("=" * 60)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Target: {MODEL_DIR}")
    print("\nModel Info:")
    print("  - Total params: 16B")
    print("  - Active params: 2B (MoE)")
    print("  - Experts: 64 (top-6)")
    print("  - Context: 32K tokens")

    # Create directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\nDownloading from ModelScope...")
    print("(This may take 30-60 minutes depending on network speed)\n")

    try:
        snapshot_download(
            MODEL_NAME,
            cache_dir=BASE_DIR,
            local_dir=MODEL_DIR,
        )

        # Calculate size
        total_size = sum(f.stat().st_size for f in MODEL_DIR.rglob('*') if f.is_file())
        size_gb = total_size / (1024**3)

        print(f"\n{'=' * 60}")
        print(f"✅ Download complete!")
        print(f"{'=' * 60}")
        print(f"Model size: {size_gb:.2f} GB")
        print(f"Location: {MODEL_DIR}")
        print(f"\nTo train with TP:")
        print(f"  torchrun --nproc_per_node=4 examples/train_tp_moe.py \\")
        print(f"    --model_path '{MODEL_DIR}' \\")
        print(f"    --tp_size 4 \\")
        print(f"    --batch_size 1")

        return True

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check network connection")
        print("2. Verify ModelScope is installed: pip install modelscope")
        print("3. Try using aria2 download script instead")
        return False

if __name__ == "__main__":
    download_deepseek_v2_lite()
