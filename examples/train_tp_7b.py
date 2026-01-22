#!/usr/bin/env python3
"""
Tensor Parallel Training on Ascend NPU using PyTorch Native TP

This script uses torch.distributed.tensor.parallel for tensor parallelism
on 8x NPU 910A system.
"""

import os
import time
import argparse
import torch
import torch_npu
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)


def setup_npu():
    """Setup NPU device and distributed"""
    # Get environment variables from torchrun
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
    parser = argparse.ArgumentParser(description="TP Training on NPU using PyTorch native TP")
    parser.add_argument("--model_path", type=str, default="/home/sd/npu_train/models/Qwen-Qwen2.5-7B-Instruct",
                        help="Model path")
    parser.add_argument("--tp_size", type=int, default=4,
                        help="Tensor parallel size (default: 4)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size per device (default: 2)")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Max sequence length (default: 2048)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs (default: 1)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (default: 1e-5)")
    parser.add_argument("--warmup_steps", type=int, default=10,
                        help="Warmup steps (default: 10)")
    parser.add_argument("--log_interval", type=int, default=5,
                        help="Log every N steps (default: 5)")
    parser.add_argument("--save_path", type=str, default="./output_tp",
                        help="Model save path")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use automatic mixed precision")
    return parser.parse_args()


def prepare_synthetic_data(args, tokenizer, rank):
    """Generate synthetic training data"""
    # Sample texts for language modeling
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

    max_samples = 100
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

    # Distributed sampler
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
        num_workers=0,
    )

    return dataloader


def get_tp_parallelize_plan(model):
    """
    Get TP parallelize plan for a HuggingFace model.

    Qwen/Llama style model has:
    - model.layers[i].self_attn:
        - q_proj, k_proj, v_proj: Column parallel
        - o_proj: Row parallel
    - model.layers[i].mlp:
        - gate_proj, up_proj: Column parallel
        - down_proj: Row parallel
    """
    plan = {}

    # Find all layers and their modules
    for name, module in model.named_modules():
        # Qwen/Llama style: .self_attn.q_proj, .self_attn.k_proj, etc.
        if name.endswith(".self_attn.q_proj"):
            plan[name] = ColwiseParallel()
        elif name.endswith(".self_attn.k_proj"):
            plan[name] = ColwiseParallel()
        elif name.endswith(".self_attn.v_proj"):
            plan[name] = ColwiseParallel()
        # Output projection: Row parallel
        elif name.endswith(".self_attn.o_proj"):
            plan[name] = RowwiseParallel()
        # MLP projections
        elif name.endswith(".mlp.gate_proj"):
            plan[name] = ColwiseParallel()
        elif name.endswith(".mlp.up_proj"):
            plan[name] = ColwiseParallel()
        elif name.endswith(".mlp.down_proj"):
            plan[name] = RowwiseParallel()
        # LM head: Column parallel (partial output for loss)
        elif name.endswith(".lm_head"):
            plan[name] = ColwiseParallel()

    return plan


def main():
    """Main training function"""
    args = get_args()

    # Setup distributed
    rank, world_size, device = setup_npu()
    args.world_size = world_size

    # Only print from rank 0
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Tensor Parallel Training with PyTorch Native TP")
        print(f"{'='*60}")
        print(f"Model: {args.model_path}")
        print(f"TP Size: {args.tp_size}")
        print(f"World Size: {world_size}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Max Length: {args.max_length}")
        print(f"Device: NPU")
        print(f"{'='*60}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in bf16 if supported, otherwise fp16
    model_dtype = torch.bfloat16 if torch.npu.is_available() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=model_dtype,
        device_map={"": device},
    )

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Expected parameters per TP rank: {total_params // args.tp_size:,}")

    # Create device mesh for tensor parallelism
    # Use "npu" as device type (PyTorch 2.5+ should support it)
    try:
        tp_mesh = init_device_mesh("npu", (args.tp_size,))
    except Exception as e:
        # Fallback to "cuda" if "npu" not supported
        print(f"[Rank {rank}] Warning: Using 'cuda' device_type for DeviceMesh: {e}")
        tp_mesh = init_device_mesh("cuda", (args.tp_size,))

    if rank == 0:
        print(f"TP Device Mesh: {tp_mesh}")
        print(f"\nApplying tensor parallelism...")

    # Apply tensor parallelism to model
    tp_plan = get_tp_parallelize_plan(model)
    model = parallelize_module(model, tp_mesh, tp_plan)

    if rank == 0:
        print(f"Tensor parallelism applied to {len(tp_plan)} layers")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # Scheduler
    num_training_steps = 100  # Simplified for testing
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Prepare data
    dataloader = prepare_synthetic_data(args, tokenizer, rank)

    # Training settings
    model.train()
    # Use NPU ShardedGradScaler for distributed training
    scaler = None
    if args.use_amp:
        try:
            scaler = torch_npu.amp.ShardedGradScaler()
            if rank == 0:
                print(f"Using NPU ShardedGradScaler for AMP")
        except Exception as e:
            if rank == 0:
                print(f"Warning: ShardedGradScaler not available: {e}")
            args.use_amp = False

    if rank == 0:
        print(f"\nStarting training...")
        if args.use_amp:
            print("AMP: Enabled")
        else:
            print("AMP: Disabled")

    # Training loop
    step = 0
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
            if rank == 0 and step % 50 == 0 and step > 0:
                save_path = os.path.join(args.save_path, f"checkpoint_step_{step}")
                os.makedirs(save_path, exist_ok=True)
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"[Rank {rank}] Saved checkpoint to {save_path}")

            step += 1
            if step >= num_training_steps:
                break

        if step >= num_training_steps:
            break

    # Final save
    if rank == 0:
        final_path = os.path.join(args.save_path, "final_model")
        os.makedirs(final_path, exist_ok=True)
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"\n[Rank {rank}] Final model saved to {final_path}")

    # Cleanup
    dist.destroy_process_group()

    if rank == 0:
        print("\nTraining completed!")


if __name__ == "__main__":
    main()
