#!/usr/bin/env python3
"""
Tensor Parallel Training on Ascend NPU using Custom NPU TP Implementation

This script uses custom NPU TP implementation based on Megatron-LM patterns.
Works when PyTorch native TP is not compatible with NPU backend.
"""

import os
import time
import argparse
import json
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
from npu_parallel.checkpoint_utils import load_tp_rank_checkpoint, save_tp_rank_checkpoint
from npu_parallel.npu_compat import compatibility_report, reset_fallback_stats, set_compat_policy


def setup_npu():
    """Setup NPU device and distributed"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize distributed with HCCL backend
    dist.init_process_group(backend="hccl")

    # Set NPU device
    torch.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")

    print(f"[Rank {rank}] NPU device: {device}")
    print(f"[Rank {rank}] World size: {world_size}")

    return rank, world_size, device


def get_args():
    parser = argparse.ArgumentParser(description="TP Training on NPU using custom TP implementation")
    parser.add_argument("--model_path", type=str, default="/home/sd/npu_train/models/Qwen-Qwen2.5-1.5B-Instruct",
                        help="Model path")
    parser.add_argument("--tp_size", type=int, default=2,
                        help="Tensor parallel size (default: 2)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size per device (default: 2)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max sequence length (default: 512)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs (default: 1)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override the default fixed training-step budget")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--warmup_steps", type=int, default=5,
                        help="Warmup steps (default: 5)")
    parser.add_argument("--log_interval", type=int, default=2,
                        help="Log every N steps (default: 2)")
    parser.add_argument("--save_path", type=str, default="./output_tp_custom",
                        help="Model save path")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from a TP checkpoint directory")
    parser.add_argument("--compat_policy", choices=["fallback", "warn", "strict"], default=None,
                        help="Compatibility fallback policy")
    parser.add_argument("--compat_report", action="store_true",
                        help="Print compatibility report before training")
    parser.add_argument("--compat_report_file", type=str, default=None,
                        help="Write post-training compatibility report JSON (rank 0 only)")
    parser.add_argument("--skip_save", action="store_true",
                        help="Skip checkpoint/model save for smoke runs")
    return parser.parse_args()


def prepare_synthetic_data(args, tokenizer, rank):
    """Generate synthetic training data"""
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. " * 10,
        "Machine learning is transforming artificial intelligence. " * 10,
        "Deep learning models can learn complex patterns. " * 10,
        "Neural networks are inspired by brain structure. " * 10,
        "Natural language processing enables text understanding. " * 10,
        "Transformers revolutionized natural language processing. " * 10,
        "Attention mechanisms focus on relevant information. " * 10,
        "Pre-trained models adapt to various tasks. " * 10,
    ]

    max_samples = 50  # Smaller dataset for testing
    texts = []
    for i in range(max_samples):
        texts.append(sample_texts[i % len(sample_texts)])

    # Tokenize
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=args.max_length,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = encodings["input_ids"]
    labels = input_ids.clone()

    # Create dataset
    dataset = torch.utils.data.TensorDataset(input_ids, labels)

    # Distributed sampler - use TP group size for proper sharding
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    return dataloader


def maybe_write_compat_report(report_path, rank, args, completed_step):
    if rank != 0 or not report_path:
        return

    report_payload = {
        "context": {
            "script": os.path.basename(__file__),
            "model_path": args.model_path,
            "tp_size": args.tp_size,
            "world_size": getattr(args, "world_size", None),
            "max_steps": args.max_steps,
            "completed_step": int(completed_step),
        },
        "report": compatibility_report(device_type="npu"),
    }

    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2, sort_keys=True)
    print(f"[Rank {rank}] Wrote compatibility report to {report_path}")


def main():
    """Main training function"""
    args = get_args()
    if args.compat_policy is not None:
        set_compat_policy(args.compat_policy)

    # Setup distributed
    rank, world_size, device = setup_npu()
    args.world_size = world_size
    if rank == 0 and args.compat_report:
        print("[Compatibility Report]")
        print(json.dumps(compatibility_report(device_type="npu"), indent=2, sort_keys=True))

    # Calculate TP rank based on world_size and tp_size
    # For pure TP (world_size == tp_size), each rank is its own TP rank
    # For hybrid TP+DDP, we'd need proper group management
    if world_size == args.tp_size:
        tp_rank = rank  # Pure TP
        use_ddp = False
    else:
        # For hybrid TP+DDP: ranks are organized as [dp0_tp0, dp0_tp1, ..., dp1_tp0, ...]
        dp_size = world_size // args.tp_size
        dp_rank = rank // args.tp_size
        tp_rank = rank % args.tp_size
        use_ddp = True

    # Only print from rank 0
    if rank == 0:
        print(f"\n{'='*60}")
        print("Tensor Parallel Training - Custom NPU TP")
        print(f"{'='*60}")
        print(f"Model: {args.model_path}")
        print(f"TP Size: {args.tp_size}")
        print(f"World Size: {world_size}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Max Length: {args.max_length}")
        print(f"Device: NPU")
        if use_ddp:
            print(f"Hybrid TP+DDP: TP={args.tp_size}, DP={dp_size}")
        print(f"{'='*60}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    # Use float32 for NPU compatibility (7B model has issues with bfloat16)
    # 1.5B model can use bfloat16, but 7B needs float32
    model_dtype = torch.float32  # Use float32 for stability
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=model_dtype,
        device_map={"": device},
        trust_remote_code=True,
    )

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Expected parameters per TP rank: {total_params // args.tp_size:,}")

    # Convert to TP using custom implementation
    if args.tp_size > 1:
        if rank == 0:
            print(f"\nApplying custom tensor parallelism (tp_size={args.tp_size})...")
        model = convert_to_tp(model, tp_size=args.tp_size, rank=tp_rank)
        model.to(device)
        if rank == 0:
            print(f"Tensor parallelism applied!")
            print(f"Rank {rank} -> TP rank {tp_rank}")
    else:
        model.to(device)

    # Setup optimizer - only on parameters requiring grad
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # Scheduler
    num_training_steps = args.max_steps if args.max_steps is not None else 50  # Simplified for testing
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Prepare data
    dataloader = prepare_synthetic_data(args, tokenizer, rank)

    start_step = 0
    if args.resume_from:
        restored_state = load_tp_rank_checkpoint(
            args.resume_from,
            model,
            rank,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_step = int(restored_state.get("step", -1)) + 1
        if rank == 0:
            print(f"Resumed TP checkpoint from {args.resume_from} at step {start_step}")

    reset_fallback_stats()

    # Training settings
    model.train()

    if rank == 0:
        print(f"\nStarting training...")
        print(f"Total steps: {num_training_steps}")
        print(f"Log interval: {args.log_interval}")

    # Training loop
    step = start_step
    start_time = time.time()
    for epoch in range(args.epochs):
        dataloader.sampler.set_epoch(epoch)

        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            # Move to device
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Sync gradients across TP ranks
            if args.tp_size > 1:
                sync_gradients_tp(model, args.tp_size)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # Logging
            if rank == 0 and step % args.log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                loss_value = loss.item()
                elapsed = time.time() - start_time
                tokens_per_sec = args.batch_size * args.max_length * (step + 1) / elapsed if step > 0 else 0
                print(f"Step {step}/{num_training_steps} | Loss: {loss_value:.4f} | LR: {lr:.2e} | Tokens/sec: {tokens_per_sec:.0f}")

            # Save checkpoint periodically
            if not args.skip_save and step % 25 == 0 and step > 0:
                save_path = os.path.join(args.save_path, f"checkpoint_step_{step}")
                save_path_rank = save_tp_rank_checkpoint(
                    save_path,
                    model,
                    rank,
                    args.tp_size,
                    tokenizer=tokenizer if rank == 0 else None,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    training_state={"step": step, "epoch": epoch},
                    extra_metadata={"source": "train_tp_custom.py", "step": step},
                )
                if rank == 0:
                    print(f"[Rank {rank}] Saved checkpoint to {save_path_rank}")

            step += 1
            if step >= num_training_steps:
                break

        if step >= num_training_steps:
            break

    if args.skip_save:
        if rank == 0:
            print("\n[Rank 0] Skipped checkpoint/model save (--skip_save)")
    else:
        final_path = os.path.join(args.save_path, "final_model")
        save_path_rank = save_tp_rank_checkpoint(
            final_path,
            model,
            rank,
            args.tp_size,
            tokenizer=tokenizer if rank == 0 else None,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state={"step": max(step - 1, 0), "epoch": args.epochs - 1, "final": True},
            extra_metadata={"source": "train_tp_custom.py", "final": True},
        )
        if rank == 0:
            print(f"\n[Rank {rank}] Final model saved to {save_path_rank}")

    maybe_write_compat_report(args.compat_report_file, rank, args, max(step - 1, -1))

    # Cleanup
    dist.destroy_process_group()

    if rank == 0:
        print("\nTraining completed!")


if __name__ == "__main__":
    main()
