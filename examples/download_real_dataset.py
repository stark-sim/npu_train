#!/usr/bin/env python3
"""
下载真实训练数据集

支持的数据集：
- WikiText-103: 500MB，适合快速验证
- SlimPajama-6B: 300GB，适合正式训练
- C4: 750GB，通用英语文本
"""

import os
import argparse
from datasets import load_dataset
from tqdm import tqdm


def download_wikitext(save_path, cache_dir=None):
    """
    下载 WikiText-103 数据集

    WikiText-103 是一个包含长期依赖的语言建模数据集
    - 大小: ~500MB
    - 样本数: ~180K 训练样本
    - 来源: 维基百科文章
    """
    print("=" * 60)
    print("下载 WikiText-103 数据集")
    print("=" * 60)

    print("Loading train split...")
    train_dataset = load_dataset(
        "wikitext",
        "wikitext-103-v1",
        split="train",
        cache_dir=cache_dir
    )
    print(f"  Train samples: {len(train_dataset):,}")

    print("Loading validation split...")
    val_dataset = load_dataset(
        "wikitext",
        "wikitext-103-v1",
        split="validation",
        cache_dir=cache_dir
    )
    print(f"  Val samples: {len(val_dataset):,}")

    print("Loading test split...")
    test_dataset = load_dataset(
        "wikitext",
        "wikitext-103-v1",
        split="test",
        cache_dir=cache_dir
    )
    print(f"  Test samples: {len(test_dataset):,}")

    # 保存到磁盘
    os.makedirs(save_path, exist_ok=True)
    train_path = os.path.join(save_path, "train")
    val_path = os.path.join(save_path, "validation")
    test_path = os.path.join(save_path, "test")

    print(f"\nSaving to {save_path}...")
    train_dataset.save_to_disk(train_path)
    print(f"  Saved train to {train_path}")

    val_dataset.save_to_disk(val_path)
    print(f"  Saved validation to {val_path}")

    test_dataset.save_to_disk(test_path)
    print(f"  Saved test to {test_path}")

    # 统计信息
    total_samples = len(train_dataset) + len(val_dataset) + len(test_dataset)
    print(f"\n总样本数: {total_samples:,}")

    return train_dataset, val_dataset, test_dataset


def download_slimpajama(save_path, cache_dir=None, num_samples=None):
    """
    下载 SlimPajama-6B 数据集

    SlimPajama 是经过清洗的 The Pile 子集
    - 大小: ~300GB (完整版)
    - 样本数: ~29B tokens
    - 来源: 多领域高质量文本
    """
    print("=" * 60)
    print("下载 SlimPajama-6B 数据集")
    print("=" * 60)

    print("Loading SlimPajama-6B (this may take a while)...")
    dataset = load_dataset(
        "cerebras/SlimPajama-6B",
        split="train",
        cache_dir=cache_dir
    )

    print(f"  Total samples: {len(dataset):,}")

    # 限制样本数（用于测试）
    if num_samples:
        print(f"  Limiting to {num_samples:,} samples...")
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    # 保存到磁盘
    os.makedirs(save_path, exist_ok=True)
    print(f"\nSaving to {save_path}...")
    dataset.save_to_disk(save_path)
    print(f"  Saved to {save_path}")

    return dataset


def download_c4(save_path, cache_dir=None, num_samples=None):
    """
    下载 C4 (Colossal Clean Crawled Corpus) 数据集

    C4 是经过清洗的网页文本数据集
    - 大小: ~750GB (完整版)
    - 样本数: 数百万
    - 来源: 清洗后的网页文本
    """
    print("=" * 60)
    print("下载 C4 数据集 (en subset)")
    print("=" * 60)

    print("Loading C4 (en)...")
    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        cache_dir=cache_dir
    )

    print(f"  Total samples: {len(dataset):,}")

    # 限制样本数（用于测试）
    if num_samples:
        print(f"  Limiting to {num_samples:,} samples...")
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    # 保存到磁盘
    os.makedirs(save_path, exist_ok=True)
    print(f"\nSaving to {save_path}...")
    dataset.save_to_disk(save_path)
    print(f"  Saved to {save_path}")

    return dataset


def load_from_disk(dataset_path):
    """从磁盘加载数据集"""
    from datasets import load_from_disk as hf_load_from_disk

    print(f"Loading dataset from {dataset_path}...")
    dataset = hf_load_from_disk(dataset_path)
    print(f"  Loaded {len(dataset):,} samples")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="下载真实训练数据集")
    parser.add_argument("--dataset", type=str, default="wikitext",
                        choices=["wikitext", "slimpajama", "c4"],
                        help="数据集名称")
    parser.add_argument("--save_path", type=str,
                        default="/home/sd/npu_train/datasets",
                        help="数据集保存路径")
    parser.add_argument("--cache_dir", type=str,
                        default="/home/sd/npu_train/datasets_cache",
                        help="HuggingFace 缓存目录")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="限制样本数量（用于测试）")
    parser.add_argument("--verify", type=str, default=None,
                        help="验证已下载的数据集路径")

    args = parser.parse_args()

    # 验证模式
    if args.verify:
        try:
            dataset = load_from_disk(args.verify)
            print(f"\n数据集验证成功!")
            print(f"  样本数: {len(dataset):,}")
            print(f"  特征: {dataset.column_names}")
            print(f"  第一个样本:")
            for key, value in dataset[0].items():
                if isinstance(value, str):
                    print(f"    {key}: {value[:100]}...")
                else:
                    print(f"    {key}: {value}")
            return
        except Exception as e:
            print(f"数据集验证失败: {e}")
            return

    # 构建完整保存路径
    if args.dataset == "wikitext":
        save_path = os.path.join(args.save_path, "wikitext-103")
        download_wikitext(save_path, args.cache_dir)
    elif args.dataset == "slimpajama":
        save_path = os.path.join(args.save_path, "slimpajama-6b")
        download_slimpajama(save_path, args.cache_dir, args.num_samples)
    elif args.dataset == "c4":
        save_path = os.path.join(args.save_path, "c4-en")
        download_c4(save_path, args.cache_dir, args.num_samples)

    print("\n" + "=" * 60)
    print("数据集下载完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
