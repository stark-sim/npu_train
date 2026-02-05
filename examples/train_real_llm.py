#!/usr/bin/env python3
"""
真实 LLM 训练脚本

解决三大问题：
1. 使用真实文本数据集（HuggingFace datasets）
2. 包含验证集，计算困惑度
3. 支持长期训练（2-3天）

支持的数据集格式：
- HuggingFace datasets (Arrow 格式)
- 自带 train/val 划分
"""

import os
import time
import math
import argparse
import json
import torch
import torch_npu
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from npu_parallel import convert_to_tp, sync_gradients_tp


class TextDataset(Dataset):
    """从预处理的文本数据创建 Dataset"""

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


class HFDataset(Dataset):
    """从 HuggingFace Arrow 格式加载 Dataset"""

    def __init__(self, dataset_path, tokenizer, max_length, text_field="text"):
        from datasets import load_from_disk
        self.dataset = load_from_disk(dataset_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_field = text_field

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx][self.text_field]

        # 跳过空文本
        if not text or text.isspace():
            text = "Empty text."

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


def compute_perplexity(model, dataloader, device, rank, tokenizer):
    """计算验证集困惑度"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
            loss = outputs.loss

            # 计算非 padding token 数量
            attention_mask = (input_ids != tokenizer.pad_token_id).float()
            num_tokens = attention_mask.sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)

    model.train()
    return avg_loss, perplexity


def save_checkpoint(model, optimizer, scheduler, step, val_loss, val_ppl, save_path, rank):
    """保存训练检查点"""
    if rank != 0:
        return

    os.makedirs(save_path, exist_ok=True)
    checkpoint_file = os.path.join(save_path, f"checkpoint_step_{step}.pt")

    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'val_ppl': val_ppl,
    }, checkpoint_file)

    # 保存最佳模型
    best_file = os.path.join(save_path, "best_model.pt")
    if not os.path.exists(best_file):
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'val_ppl': val_ppl,
        }, best_file)
    else:
        # 加载当前最佳，比较困惑度
        best_checkpoint = torch.load(best_file, map_location='cpu')
        if val_ppl < best_checkpoint['val_ppl']:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_ppl': val_ppl,
            }, best_file)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, rank):
    """加载训练检查点以恢复训练"""
    if not os.path.exists(checkpoint_path):
        if rank == 0:
            print(f"检查点不存在: {checkpoint_path}")
        return 0, None, None

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_step = checkpoint.get('step', 0) + 1

    if rank == 0:
        print(f"从检查点恢复: step={start_step}")
        if 'val_loss' in checkpoint:
            print(f"  之前验证损失: {checkpoint['val_loss']:.4f}")
        if 'val_ppl' in checkpoint:
            print(f"  之前困惑度: {checkpoint['val_ppl']:.2f}")

    return start_step, checkpoint.get('val_loss', None), checkpoint.get('val_ppl', None)


def main():
    parser = argparse.ArgumentParser(description="真实 LLM 训练")
    parser.add_argument("--model_path", type=str,
                        default="/home/sd/npu_train/models/deepseek-ai-DeepSeek-V2-Lite",
                        help="模型路径")
    parser.add_argument("--dataset_path", type=str,
                        default="/home/sd/npu_train/datasets/wikitext-103",
                        help="数据集路径 (train/ 目录)")
    parser.add_argument("--val_dataset_path", type=str, default=None,
                        help="验证集路径 (可选，默认使用 dataset_path/validation)")
    parser.add_argument("--output_path", type=str,
                        default="./output_real_llm",
                        help="输出路径")
    parser.add_argument("--tp_size", type=int, default=4,
                        help="张量并行大小")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="批次大小 (每卡)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--num_steps", type=int, default=200000,
                        help="训练步数")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="学习率")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                        help="预热步数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数")
    parser.add_argument("--eval_interval", type=int, default=5000,
                        help="评估间隔（步数）")
    parser.add_argument("--save_interval", type=int, default=10000,
                        help="保存间隔（步数）")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="从检查点恢复训练")
    parser.add_argument("--text_field", type=str, default="text",
                        help="数据集中的文本字段名")
    return parser.parse_args()


args = main()

# Setup distributed
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", args.tp_size))
local_rank = int(os.environ.get("LOCAL_RANK", rank))

# HCCL timeout (extended for long training)
os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "7200")
os.environ.setdefault("HCCL_EXEC_TIMEOUT", "7200")
os.environ.setdefault("TORCH_NPU_ALLOC_CONF", "max_split_size_mb:128")

dist.init_process_group(backend="hccl")
torch.npu.set_device(local_rank)
device = torch.device(f"npu:{local_rank}")

# Print config
if rank == 0:
    print("=" * 70)
    print("真实 LLM 训练")
    print("=" * 70)
    print(f"模型: {args.model_path}")
    print(f"数据集: {args.dataset_path}")
    print(f"TP Size: {args.tp_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Gradient Accumulation: {args.gradient_accumulation_steps}")
    print(f"有效 Batch Size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Max Length: {args.max_length}")
    print(f"Num Steps: {args.num_steps}")
    print(f"Learning Rate: {args.lr}")
    print(f"Warmup Steps: {args.warmup_steps}")
    print("=" * 70)

# Load tokenizer
if rank == 0:
    print("\n[1/7] 加载 tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if rank == 0:
    print(f"  Vocab size: {len(tokenizer)}")

dist.barrier()

# Load model
if rank == 0:
    print("[2/7] 加载模型...")

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

if rank == 0:
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数: {total_params:,} ({total_params/1e9:.2f}B)")

dist.barrier()

# TP conversion
if rank == 0:
    print("[3/7] TP 转换...")

tp_rank = rank % args.tp_size  # 支持 TP+DDP
model = convert_to_tp(model, tp_size=args.tp_size, rank=tp_rank)
model.to(device)
model.config.use_cache = False

if rank == 0:
    print("  TP 转换完成")

dist.barrier()

# Load datasets
if rank == 0:
    print("[4/7] 加载数据集...")

train_path = os.path.join(args.dataset_path, "train")
val_path = args.val_dataset_path or os.path.join(args.dataset_path, "validation")

train_dataset = HFDataset(train_path, tokenizer, args.max_length, args.text_field)
val_dataset = HFDataset(val_path, tokenizer, args.max_length, args.text_field)

if rank == 0:
    print(f"  训练集: {len(train_dataset):,} 样本")
    print(f"  验证集: {len(val_dataset):,} 样本")

dist.barrier()

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

# Optimizer and scheduler
if rank == 0:
    print("[5/7] 设置优化器...")

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.01,
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=args.warmup_steps,
    num_training_steps=args.num_steps,
)

if rank == 0:
    print(f"  优化器: AdamW (lr={args.lr}, weight_decay=0.01)")
    print(f"  调度器: LinearLR (warmup={args.warmup_steps} steps)")

dist.barrier()

# Resume from checkpoint if specified
start_step = 0
best_ppl = float('inf')

if args.resume_from:
    if rank == 0:
        print("[6/7] 从检查点恢复...")
    start_step, _, best_ppl = load_checkpoint(
        model, optimizer, scheduler, args.resume_from, rank
    )
    dist.barrier()
else:
    # Initial evaluation
    if rank == 0:
        print("[6/7] 初始评估...")

    initial_val_loss, initial_ppl = compute_perplexity(
        model, val_loader, device, rank, tokenizer
    )

    if rank == 0:
        print(f"  初始 Val Loss: {initial_val_loss:.4f}")
        print(f"  初始困惑度: {initial_ppl:.2f}")
        best_ppl = initial_ppl

    dist.barrier()

# Create output directory
if rank == 0:
    os.makedirs(args.output_path, exist_ok=True)

    # Save training config
    config = {
        'model_path': args.model_path,
        'dataset_path': args.dataset_path,
        'tp_size': args.tp_size,
        'batch_size': args.batch_size,
        'max_length': args.max_length,
        'num_steps': args.num_steps,
        'lr': args.lr,
        'warmup_steps': args.warmup_steps,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
    }

    with open(os.path.join(args.output_path, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

dist.barrier()

# Training loop
if rank == 0:
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)

model.train()
step = start_step
start_time = time.time()
accumulation_step = 0

import itertools
train_iterator = itertools.cycle(train_loader)

while step < args.num_steps:
    batch = next(train_iterator)
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    # Forward pass
    outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
    loss = outputs.loss / args.gradient_accumulation_steps

    # Backward pass
    loss.backward()

    accumulation_step += 1

    # Update weights
    if accumulation_step >= args.gradient_accumulation_steps:
        sync_gradients_tp(model, args.tp_size)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        accumulation_step = 0

        # Logging
        if rank == 0 and step % 100 == 0:
            lr = scheduler.get_last_lr()[0]
            loss_val = loss.item() * args.gradient_accumulation_steps
            elapsed = time.time() - start_time
            tokens_per_sec = (step * args.batch_size * args.max_length) / elapsed
            eta_hours = ((args.num_steps - step) / (step / (elapsed / 3600))) if step > 0 else 0

            print(f"Step {step}/{args.num_steps} | "
                  f"Loss: {loss_val:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Tok/s: {tokens_per_sec:.0f} | "
                  f"ETA: {eta_hours:.1f}h")

        # Evaluation (every eval_interval steps)
        if (step + 1) % args.eval_interval == 0:
            val_loss, val_ppl = compute_perplexity(
                model, val_loader, device, rank, tokenizer
            )

            if rank == 0:
                print(f"\n{'='*50}")
                print(f"验证 @ Step {step}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val Perplexity: {val_ppl:.2f}")
                print(f"{'='*50}\n")

            # Save best model if improved
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                save_checkpoint(model, optimizer, scheduler, step,
                              val_loss, val_ppl, args.output_path, rank, is_best=True)
                if rank == 0:
                    print(f"  [最佳] 困惑度: {val_ppl:.2f}")

            # Also save periodic checkpoint at eval intervals for resuming
            save_checkpoint(model, optimizer, scheduler, step,
                          val_loss, val_ppl, args.output_path, rank, is_best=False)

        # Periodic checkpoint (for resuming, at non-eval steps)
        # This catches cases where save_interval is not a multiple of eval_interval
        elif (step + 1) % args.save_interval == 0 and (step + 1) < args.num_steps:
            save_checkpoint(model, optimizer, scheduler, step,
                          None, None, args.output_path, rank, is_best=False)
            if rank == 0:
                print(f"  保存检查点: step_{step}")

        step += 1

# Final evaluation
if rank == 0:
    print("\n" + "=" * 70)
    print("训练完成 - 最终评估")
    print("=" * 70)

final_val_loss, final_ppl = compute_perplexity(
    model, val_loader, device, rank, tokenizer
)

if rank == 0:
    print(f"最终 Val Loss: {final_val_loss:.4f}")
    print(f"最终困惑度: {final_ppl:.2f}")
    print(f"最佳困惑度: {best_ppl:.2f}")
    print(f"总训练时间: {time.time() - start_time:.1f}s ({(time.time() - start_time)/3600:.2f}h)")

    # Save final model
    final_path = os.path.join(args.output_path, "final_model.pt")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'val_loss': final_val_loss,
        'val_ppl': final_ppl,
    }, final_path)
    print(f"最终模型已保存: {final_path}")

dist.destroy_process_group()

if rank == 0:
    print("\n[Done] 训练完成!")


if __name__ == "__main__":
    main()
