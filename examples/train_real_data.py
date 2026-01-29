#!/usr/bin/env python3
"""
DeepSeek-V2-Lite 真实数据训练脚本

使用真实的文本数据集，包含：
- 训练集和验证集
- 困惑度计算
- 更完整的训练循环
- 模型检查点保存
"""

import os
import time
import argparse
import torch
import torch_npu
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from npu_parallel import convert_to_tp, sync_gradients_tp


class TextDataset(Dataset):
    """简单的文本数据集"""
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "labels": encodings["input_ids"].squeeze(0),
        }


def prepare_real_data(tokenizer, max_length=256, num_train=1000, num_val=100):
    """
    准备真实的训练数据

    使用简化的样本数据（避免下载大文件）
    """
    # 更多样化的训练样本
    train_texts = [
        "The quick brown fox jumps over the lazy dog. " * 5,
        "Machine learning is transforming artificial intelligence and computer science. " * 5,
        "Natural language processing enables computers to understand human language. " * 5,
        "Deep learning models have achieved remarkable results in various tasks. " * 5,
        "The transformer architecture revolutionized natural language processing. " * 5,
        "Neural networks can learn complex patterns from large amounts of data. " * 5,
        "Training large language models requires significant computational resources. " * 5,
        "Tensor parallelism is a technique for distributing models across multiple devices. " * 5,
        "Mixture of Experts models enable efficient scaling of model parameters. " * 5,
        "The attention mechanism allows models to focus on relevant parts of the input. " * 5,
        "Gradient descent is a fundamental optimization algorithm for training neural networks. " * 5,
        "Language models are trained on vast amounts of text from the internet. " * 5,
        "The performance of a model depends on its architecture and training data quality. " * 5,
        "Modern NPU hardware accelerates the training of large deep learning models. " * 5,
        "Distributed training techniques enable scaling to larger models and datasets. " * 5,
    ]

    # 验证集（不同的文本）
    val_texts = [
        "Artificial intelligence is advancing rapidly in recent years. " * 5,
        "Computer vision and natural language processing are key AI applications. " * 5,
        "Deep learning has transformed the field of machine learning research. " * 5,
        "Language models generate human-like text based on patterns they learned. " * 5,
    ]

    # 重复到足够的数量
    train_texts = train_texts * (num_train // len(train_texts) + 1)
    train_texts = train_texts[:num_train]
    val_texts = val_texts * (num_val // len(val_texts) + 1)
    val_texts = val_texts[:num_val]

    train_dataset = TextDataset(train_texts, tokenizer, max_length)
    val_dataset = TextDataset(val_texts, tokenizer, max_length)

    return train_dataset, val_dataset


def compute_perplexity(model, dataloader, device, rank):
    """计算困惑度"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # 只计算非padding位置的loss
            outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
            loss = outputs.loss

            # 计算token数量（排除padding）
            attention_mask = (input_ids != tokenizer.pad_token_id).float()
            num_tokens = attention_mask.sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    # 计算平均loss和困惑度
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    model.train()
    return avg_loss, perplexity


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description="DeepSeek-V2-Lite 真实数据训练")
    parser.add_argument("--model_path", type=str,
                        default="/home/sd/npu_train/models/deepseek-ai-DeepSeek-V2-Lite",
                        help="模型路径")
    parser.add_argument("--tp_size", type=int, default=4,
                        help="张量并行大小")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="批次大小")
    parser.add_argument("--max_length", type=int, default=128,
                        help="最大序列长度")
    parser.add_argument("--num_train_samples", type=int, default=1000,
                        help="训练样本数")
    parser.add_argument("--num_val_samples", type=int, default=100,
                        help="验证样本数")
    parser.add_argument("--num_steps", type=int, default=500,
                        help="训练步数")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--warmup_steps", type=int, default=50,
                        help="预热步数")
    parser.add_argument("--eval_interval", type=int, default=100,
                        help="评估间隔（步数）")
    parser.add_argument("--save_interval", type=int, default=200,
                        help="保存间隔（步数）")
    parser.add_argument("--save_path", type=str, default="./output_deepseek_real",
                        help="模型保存路径")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="数据类型")
    return parser.parse_args()


# 获取参数
args = main()

# Setup distributed
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", args.tp_size))
local_rank = int(os.environ.get("LOCAL_RANK", rank))

# HCCL timeout
os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "7200")
os.environ.setdefault("HCCL_EXEC_TIMEOUT", "7200")
os.environ.setdefault("TORCH_NPU_ALLOC_CONF", "max_split_size_mb:128")

dist.init_process_group(backend="hccl")
torch.npu.set_device(local_rank)
device = torch.device(f"npu:{local_rank}")

if rank == 0:
    print("=" * 60)
    print("DeepSeek-V2-Lite 真实数据训练")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"TP Size: {args.tp_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Length: {args.max_length}")
    print(f"Train Samples: {args.num_train_samples}")
    print(f"Val Samples: {args.num_val_samples}")
    print(f"Num Steps: {args.num_steps}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 60)

# 加载tokenizer
if rank == 0:
    print("\n[1/6] 加载tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
dist.barrier()

# 加载模型
if rank == 0:
    print("[2/6] 加载模型...")

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

if rank == 0:
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数: {total_params:,}")
dist.barrier()

# TP转换
if rank == 0:
    print("[3/6] TP转换...")

tp_rank = rank
model = convert_to_tp(model, tp_size=args.tp_size, rank=tp_rank)
model.to(device)
model.config.use_cache = False

if rank == 0:
    print("  TP转换完成")
dist.barrier()

# 准备数据
if rank == 0:
    print("[4/6] 准备数据...")

train_dataset, val_dataset = prepare_real_data(
    tokenizer,
    max_length=args.max_length,
    num_train=args.num_train_samples,
    num_val=args.num_val_samples,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

if rank == 0:
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
dist.barrier()

# 优化器和调度器
if rank == 0:
    print("[5/6] 设置优化器...")

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr,
    betas=(0.9, 0.95),
    eps=1e-8,
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=args.warmup_steps,
    num_training_steps=args.num_steps,
)

if rank == 0:
    print(f"  优化器: AdamW (lr={args.lr})")
    print(f"  调度器: LinearLR with warmup ({args.warmup_steps} steps)")
dist.barrier()

# 创建保存目录
if rank == 0:
    os.makedirs(args.save_path, exist_ok=True)
dist.barrier()

# 初始评估
if rank == 0:
    print("[6/6] 初始评估...")

initial_val_loss, initial_ppl = compute_perplexity(model, val_loader, device, rank)
if rank == 0:
    print(f"  初始 Val Loss: {initial_val_loss:.4f}")
    print(f"  初始困惑度: {initial_ppl:.2f}")
dist.barrier()

# 训练循环
if rank == 0:
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

model.train()
step = 0
start_time = time.time()
best_ppl = float('inf')

import itertools
train_iterator = itertools.cycle(train_loader)

while step < args.num_steps:
    input_ids = next(train_iterator)["input_ids"].to(device)
    labels = next(train_iterator)["labels"].to(device)

    # Forward pass
    outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
    loss = outputs.loss

    # Backward pass
    loss.backward()
    sync_gradients_tp(model, args.tp_size)
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    # Logging
    if rank == 0 and step % 10 == 0:
        lr = scheduler.get_last_lr()[0]
        loss_val = loss.item()
        elapsed = time.time() - start_time
        print(f"Step {step}/{args.num_steps} | Loss: {loss_val:.4f} | "
              f"LR: {lr:.2e} | Time: {elapsed:.1f}s | "
              f"Tokens: {step * args.batch_size * args.max_length}")

    # 验证
    if (step + 1) % args.eval_interval == 0:
        val_loss, val_ppl = compute_perplexity(model, val_loader, device, rank)
        if rank == 0:
            print(f"\n--- 验证 (Step {step}) ---")
            print(f"Val Loss: {val_loss:.4f} | Val Perplexity: {val_ppl:.2f}")

        # 保存最佳模型
        if rank == 0 and val_ppl < best_ppl:
            best_ppl = val_ppl
            save_path = os.path.join(args.save_path, "best_model.pt")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_ppl': val_ppl,
            }, save_path)
            print(f"  保存最佳模型 (困惑度: {val_ppl:.2f})")

    # 检查点保存
    if (step + 1) % args.save_interval == 0 and (step + 1) < args.num_steps:
        if rank == 0:
            save_path = os.path.join(args.save_path, f"checkpoint_step_{step}.pt")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
            print(f"  保存检查点: {save_path}")

    step += 1

# 最终评估
if rank == 0:
    print("\n" + "=" * 60)
    print("训练完成 - 最终评估")
    print("=" * 60)

final_val_loss, final_ppl = compute_perplexity(model, val_loader, device, rank)

if rank == 0:
    print(f"最终 Val Loss: {final_val_loss:.4f}")
    print(f"最终困惑度: {final_ppl:.2f}")
    print(f"最佳困惑度: {best_ppl:.2f}")
    print(f"总训练时间: {time.time() - start_time:.1f}s")

dist.destroy_process_group()

if rank == 0:
    print("\n[Done] 训练完成!")


if __name__ == "__main__":
    main()
