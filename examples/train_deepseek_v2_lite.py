#!/usr/bin/env python3
"""
Simplified MoE Training Script for DeepSeek-V2-Lite on NPU

This script uses aggressive optimizations to handle the large DeepSeek-V2-Lite model:
- Progressive model loading to avoid OOM
- Simplified MoE forward pass avoiding NPU-incompatible operations
- Extended HCCL timeouts for compilation
- Small batch size and sequence length for testing
- Compilation warmup to avoid first-step timeout

Usage:
    torchrun --nproc_per_node=4 examples/train_deepseek_v2_lite.py \\
        --model_path "/path/to/deepseek-v2-lite" \\
        --tp_size 4
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from npu_parallel import convert_to_tp, sync_gradients_tp


def setup_npu():
    """Setup NPU device and distributed with extended timeouts"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Extended HCCL timeout for large models (DeepSeek-V2-Lite is 16B params)
    os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "7200")  # 2 hours
    os.environ.setdefault("HCCL_EXEC_TIMEOUT", "7200")     # 2 hours

    # NPU optimization flags
    os.environ.setdefault("TORCH_NPU_ENABLE_COMGR", "0")
    os.environ.setdefault("TORCH_NPU_ALLOC_CONF", "max_split_size_mb:128")
    os.environ.setdefault("NPU_FUSION_ENABLE", "1")

    # Initialize distributed
    dist.init_process_group(backend="hccl")

    # Set NPU device
    torch.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")

    if rank == 0:
        print(f"[Setup] NPU device: {device}")
        print(f"[Setup] World size: {world_size}")
        print(f"[Setup] HCCL timeout: 7200s")

    return rank, world_size, device


def get_args():
    parser = argparse.ArgumentParser(description="DeepSeek-V2-Lite MoE TP Training on NPU")
    parser.add_argument("--model_path", type=str,
                        default="/home/sd/npu_train/models/deepseek-ai/DeepSeek-V2-Lite",
                        help="Path to DeepSeek-V2-Lite model")
    parser.add_argument("--tp_size", type=int, default=4,
                        help="Tensor parallel size (default: 4)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per device (default: 1)")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Max sequence length (default: 256, reduced for testing)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs (default: 1)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--warmup_steps", type=int, default=5,
                        help="Warmup steps (default: 5)")
    parser.add_argument("--aux_loss_coef", type=float, default=0.01,
                        help="Auxiliary loss coefficient (default: 0.01)")
    parser.add_argument("--log_interval", type=int, default=1,
                        help="Log every N steps (default: 1)")
    parser.add_argument("--save_path", type=str, default="./output_deepseek_moe",
                        help="Model save path")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Model dtype (default: float32 for stability)")
    return parser.parse_args()


def prepare_synthetic_data(args, tokenizer, rank):
    """Generate minimal synthetic training data"""
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. " * 20,
        "Machine learning transforms artificial intelligence. " * 20,
    ]

    max_samples = 20  # Very small for quick testing
    texts = [sample_texts[i % len(sample_texts)] for i in range(max_samples)]

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=args.max_length,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = encodings["input_ids"]
    labels = input_ids.clone()

    dataset = torch.utils.data.TensorDataset(input_ids, labels)
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
        num_workers=2,
        pin_memory=True,
    )

    return dataloader


def warmup_compilation(model, device, tokenizer, rank):
    """
    Trigger NPU JIT compilation with small inputs before actual training.

    This prevents timeout during the first training step when compilation happens.
    """
    if rank == 0:
        print("\n[Warmup] Triggering NPU JIT compilation...")

    model.eval()
    try:
        with torch.no_grad():
            # Very small input for quick compilation
            warmup_ids = torch.randint(0, tokenizer.vocab_size, (1, 4), device=device)
            warmup_labels = warmup_ids.clone()

            # Run forward pass
            outputs = model(input_ids=warmup_ids, labels=warmup_labels, use_cache=False)
            loss = outputs.loss

            # Run backward pass
            loss.backward()

        if rank == 0:
            print("[Warmup] Compilation warmup complete!")

    except Exception as e:
        if rank == 0:
            print(f"[Warmup] Warning: Warmup failed with {type(e).__name__}: {e}")
            print("[Warmup] Continuing without warmup...")

    model.train()


def main():
    """Main training function"""
    args = get_args()

    # Setup distributed
    rank, world_size, device = setup_npu()

    # Calculate TP rank
    tp_rank = rank if world_size == args.tp_size else rank % args.tp_size

    # Print config
    if rank == 0:
        print(f"\n{'='*60}")
        print("DeepSeek-V2-Lite MoE TP Training")
        print(f"{'='*60}")
        print(f"Model: {args.model_path}")
        print(f"TP Size: {args.tp_size}")
        print(f"World Size: {world_size}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Max Length: {args.max_length}")
        print(f"Data Type: {args.dtype}")
        print(f"{'='*60}\n")

    # Load tokenizer
    if rank == 0:
        print("[Load] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    model_dtype = dtype_map[args.dtype]

    # Load model on CPU first
    if rank == 0:
        print(f"[Load] Loading model on CPU (dtype={args.dtype})...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=model_dtype,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[Load] Total parameters: {total_params:,}")
        print(f"[Load] Parameters per TP rank: {total_params // args.tp_size:,}")

    # Convert to TP
    if args.tp_size > 1:
        if rank == 0:
            print(f"\n[TP] Converting model to tensor parallelism (tp_size={args.tp_size})...")

        model = convert_to_tp(model, tp_size=args.tp_size, rank=tp_rank)

        if rank == 0:
            print(f"[TP] Conversion complete!")
            if getattr(model, '_is_moe', False):
                print(f"[TP] MoE architecture detected and converted!")

    # Move to device
    if rank == 0:
        print(f"\n[Device] Moving model to NPU...")

    model.to(device)

    # Disable cache for training
    model.config.use_cache = False

    # Compilation warmup
    warmup_compilation(model, device, tokenizer, rank)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # Scheduler
    num_training_steps = 20  # Very small for testing
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Prepare data
    dataloader = prepare_synthetic_data(args, tokenizer, rank)

    # Training
    model.train()

    if rank == 0:
        print(f"\n[Train] Starting training ({num_training_steps} steps)...")

    step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        dataloader.sampler.set_epoch(epoch)

        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
            loss = outputs.loss

            # Collect auxiliary losses from MoE layers
            aux_loss = torch.zeros(1, device=device)
            for module in model.modules():
                if hasattr(module, 'last_aux_loss'):
                    aux_loss += module.last_aux_loss

            # Combine losses
            total_loss = loss + args.aux_loss_coef * aux_loss

            # Backward pass
            total_loss.backward()

            # Sync gradients
            if args.tp_size > 1:
                sync_gradients_tp(model, args.tp_size)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # Logging
            if rank == 0 and step % args.log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                loss_val = loss.item()
                aux_val = aux_loss.item()
                elapsed = time.time() - start_time
                print(f"Step {step}/{num_training_steps} | "
                      f"Loss: {loss_val:.4f} | Aux: {aux_val:.6f} | "
                      f"LR: {lr:.2e} | Time: {elapsed:.1f}s")

            step += 1
            if step >= num_training_steps:
                break

        if step >= num_training_steps:
            break

    # Cleanup
    dist.destroy_process_group()

    if rank == 0:
        print("\n[Done] Training completed successfully!")


if __name__ == "__main__":
    main()
