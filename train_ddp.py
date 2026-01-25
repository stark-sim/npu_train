#!/usr/bin/env python3
"""
Multi-card distributed training on Ascend NPU using DDP (Distributed Data Parallel)
Uses all 8 NPUs for data parallelism
"""

import os
import time
import argparse
import torch
import torch.distributed as dist
import torch_npu
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

# Set CANN environment
os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "1200")


def setup_distributed():
    """Initialize distributed training"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    # Initialize process group
    if world_size > 1:
        backend = "hccl"  # Use HCCL for NPU
        dist.init_process_group(backend=backend)
        torch_npu.set_device(local_rank)
    else:
        torch_npu.set_device(0)

    return local_rank, world_size, rank, torch.device(f"npu:{local_rank}")


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_args():
    parser = argparse.ArgumentParser(description="DDP Training on NPU")
    parser.add_argument("--model_name", type=str, default="/home/sd/npu_train/models/Qwen-Qwen2.5-1.5B-Instruct",
                        help="Model path")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per device (default: 8)")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Max sequence length (default: 256)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs (default: 3)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Warmup steps (default: 100)")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log every N steps (default: 10)")
    parser.add_argument("--save_path", type=str, default="./output_ddp",
                        help="Model save path (default: ./output_ddp)")
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="Max training samples (default: 10000)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory")
    return parser.parse_args()


def prepare_dataset(args, tokenizer, rank, world_size):
    """Prepare dataset with DistributedSampler"""
    print(f"[Rank {rank}] Preparing dataset with {args.max_samples} samples")

    # Synthetic training data
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. " * 10,
        "Machine learning is a subset of artificial intelligence. " * 10,
        "Deep learning models can learn complex patterns from data. " * 10,
        "Neural networks are inspired by the human brain structure. " * 10,
        "Natural language processing enables computers to understand text. " * 10,
        "Transformers have revolutionized the field of NLP. " * 10,
        "Attention mechanisms allow models to focus on relevant parts. " * 10,
        "Pre-trained language models can be fine-tuned for various tasks. " * 10,
    ]

    # Create dataset with repeated samples
    texts = [sample_texts[i % len(sample_texts)] for i in range(args.max_samples)]

    # Tokenize all texts
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=args.max_length,
        padding="max_length",
        return_tensors="pt",
    )

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        def __len__(self):
            return len(self.encodings["input_ids"])
        def __getitem__(self, idx):
            return {key: val[idx] for key, val in self.encodings.items()}

    dataset = SimpleDataset(tokenized)

    # Use DistributedSampler for DDP
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    return dataset, sampler


def collate_fn(batch):
    """Custom collate function"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
    }


def train(args, local_rank, world_size, rank, device):
    """Main training function"""
    print(f"[Rank {rank}] Starting training, world_size={world_size}, device={device}")

    # Load tokenizer and model
    print(f"[Rank {rank}] Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Enable gradient checkpointing if requested (before DDP wrap)
    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if rank == 0:
                print("Gradient checkpointing enabled")
        else:
            if rank == 0:
                print("Warning: Model does not support gradient_checkpointing_enable()")

    # Wrap model with DDP
    model = model.to(device)
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,  # Assumes all parameters are used
            bucket_cap_mb=25,  # Tune for NPU communication
        )

    # Print model info (only rank 0)
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Rank {rank}] Total parameters: {total_params:,}")
        print(f"[Rank {rank}] Trainable parameters: {trainable_params:,}")

    # Prepare dataset
    train_dataset, sampler = prepare_dataset(args, tokenizer, rank, world_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    if rank == 0:
        print(f"\n{'='*50}")
        print(f"DDP Training Configuration:")
        print(f"  World size: {world_size}")
        print(f"  Batch size per device: {args.batch_size}")
        print(f"  Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Total steps: {total_steps}")
        print(f"  Learning rate: {args.lr}")
        print(f"{'='*50}\n")

    # Training loop
    model.train()
    global_step = 0
    total_loss = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        epoch_loss = 0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            # Move batch to NPU
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Logging (only rank 0)
                if rank == 0 and global_step % args.log_interval == 0:
                    avg_loss = total_loss / global_step
                    elapsed = time.time() - start_time
                    samples_per_sec = (global_step * args.batch_size * world_size) / elapsed

                    print(f"Epoch {epoch+1}/{args.epochs} | "
                          f"Step {step+1}/{len(train_loader)} | "
                          f"Loss: {loss.item() * args.gradient_accumulation_steps:.4f} | "
                          f"Avg Loss: {avg_loss:.4f} | "
                          f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                          f"Speed: {samples_per_sec:.1f} samples/s")

            total_loss += loss.item() * args.gradient_accumulation_steps
            epoch_loss += loss.item() * args.gradient_accumulation_steps

        # Epoch summary
        if rank == 0:
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"\n>>> Epoch {epoch+1} completed in {epoch_time:.1f}s | "
                  f"Avg Loss: {avg_epoch_loss:.4f}\n")

    # Training complete
    if rank == 0:
        total_time = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"Training completed!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Final avg loss: {total_loss/global_step:.4f}")
        print(f"{'='*50}\n")

        # Save model (only rank 0)
        os.makedirs(args.save_path, exist_ok=True)
        unwrapped_model = model.module if hasattr(model, 'module') else model
        unwrapped_model.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)
        print(f"[Rank {rank}] Model saved to {args.save_path}")

    return model


if __name__ == "__main__":
    args = get_args()
    local_rank, world_size, rank, device = setup_distributed()
    try:
        train(args, local_rank, world_size, rank, device)
    finally:
        cleanup_distributed()
