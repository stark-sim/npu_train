#!/usr/bin/env python3
"""
Simple Tensor Parallel Training - bfloat16 without master params
"""

import os
import time
import argparse
import torch
import torch_npu
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

# Import custom NPU TP implementation
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from npu_parallel import convert_to_tp, sync_gradients_tp


def setup_npu():
    """Setup NPU device and distributed"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    dist.init_process_group(backend="hccl")
    torch.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")

    print(f"[Rank {rank}] NPU device: {device}")
    print(f"[Rank {rank}] World size: {world_size}")

    return rank, world_size, device


def get_args():
    parser = argparse.ArgumentParser(description="Simple TP Training")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tp_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--save_path", type=str, default="./output_tp_simple")
    return parser.parse_args()


def main():
    args = get_args()
    rank, world_size, device = setup_npu()
    tp_rank = rank if world_size == args.tp_size else rank % args.tp_size

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"TP Training: {args.model_path}")
        print(f"TP Size: {args.tp_size}, Batch: {args.batch_size}, Seq Len: {args.max_length}")
        print(f"{'='*60}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with bfloat16
    if rank == 0:
        print(f"Loading model with bfloat16...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

    # Convert to TP
    if args.tp_size > 1:
        if rank == 0:
            print(f"Converting to TP (tp_size={args.tp_size})...")
        model = convert_to_tp(model, tp_size=args.tp_size, rank=tp_rank)

    # Use SGD optimizer - simpler and less memory than AdamW
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.9,
    )

    # Create simple data
    texts = ["Hello world " * 50] * 20
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=args.max_length,
        padding="max_length",
        return_tensors="pt",
    )
    dataset = torch.utils.data.TensorDataset(encodings["input_ids"], encodings["input_ids"].clone())

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    model.train()

    if rank == 0:
        print(f"\nStarting training ({args.steps} steps)...\n")

    step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        dataloader.sampler.set_epoch(epoch)

        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            # Backward
            loss.backward()

            # Sync gradients
            if args.tp_size > 1:
                sync_gradients_tp(model, args.tp_size)

            optimizer.step()
            optimizer.zero_grad()

            # Logging
            if rank == 0 and step % 2 == 0:
                elapsed = time.time() - start_time
                print(f"Step {step}/{args.steps} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")

            step += 1
            if step >= args.steps:
                break

        if step >= args.steps:
            break

    dist.destroy_process_group()

    if rank == 0:
        print(f"\nTraining completed in {time.time() - start_time:.1f}s!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
