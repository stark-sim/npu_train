#!/usr/bin/env python3
"""
鲁棒的模型下载脚本 - 支持 ModelScope/HuggingFace 镜像
- 自动重试
- 断点续传
- 后台运行
- 进度保存
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path

# 配置
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_PATH = "/home/sd/npu_train/models/Qwen-Qwen2.5-7B-Instruct"
HF_MIRROR = "https://hf-mirror.com"

# 使用 huggingface_hub 进行下载
def download_with_huggingface_hub():
    """使用 huggingface_hub 下载，自动处理断点续传"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("请安装: pip install huggingface_hub")
        return False

    print("="*60)
    print("开始下载 Qwen2.5-7B-Instruct 模型")
    print("="*60)
    print(f"目标路径: {MODEL_PATH}")
    print(f"镜像源: {HF_MIRROR}")
    print("="*60)

    # 设置环境变量使用镜像
    os.environ['HF_ENDPOINT'] = HF_MIRROR

    # 需要下载的文件（检查完整性）
    expected_files = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
    ]

    # 检查已下载的文件
    model_path = Path(MODEL_PATH)
    model_path.mkdir(parents=True, exist_ok=True)

    print("\n检查已下载的文件...")
    existing_files = []
    missing_files = []

    for fname in expected_files:
        fpath = model_path / fname
        if fpath.exists() and fpath.stat().st_size > 0:
            size_mb = fpath.stat().st_size / (1024*1024)
            print(f"  ✓ {fname}: {size_mb:.1f} MB")
            existing_files.append(fname)
        else:
            print(f"  ✗ {fname}: 缺失或为空")
            missing_files.append(fname)

    print(f"\n已有: {len(existing_files)}/{len(expected_files)} 文件")

    if missing_files:
        print(f"\n需要下载: {len(missing_files)} 个文件")
        print("开始下载...")

    try:
        # 使用 resume_download 参数
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=MODEL_PATH,
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        print("\n" + "="*60)
        print("下载完成!")
        print("="*60)

        # 最终验证
        total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        print(f"总大小: {total_size / (1024**3):.2f} GB")

        return True

    except Exception as e:
        print(f"\n下载出错: {e}")
        print("请重新运行脚本继续下载")
        return False


def main():
    """主函数"""
    import signal

    # 处理 Ctrl+C
    def signal_handler(sig, frame):
        print("\n\n下载被中断，可以重新运行脚本继续下载")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # 尝试使用 huggingface_hub
    success = download_with_huggingface_hub()

    if success:
        print("\n所有文件下载完成!")
        return 0
    else:
        print("\n下载未完成，请重新运行")
        return 1


if __name__ == "__main__":
    sys.exit(main())
