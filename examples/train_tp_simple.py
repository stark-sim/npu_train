#!/usr/bin/env python3
"""
Simple TP Training Test on Ascend NPU
Minimal test to verify TP works on NPU with FP16
"""

import os
import time
import torch
import torch_npu
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import custom NPU TP implementation
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from npu_parallel import convert_to_tp, sync_gradients_tp


def setup_npu():
    """Setup NPU device and distributed"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize distributed with HCCL backend
    dist.init_process_group(backend="hccl")

    # Set NPU device
    torch.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")

    print(f"[Rank {rank}] NPU device: {device}, World size: {world_size}")

    return rank, world_size, device


def main():
    """Main training function"""
    # Setup distributed
    rank, world_size, device = setup_npu()
    tp_size = 4  # Fixed for this test

    # Calculate TP rank
    if world_size == tp_size:
        tp_rank = rank  # Pure TP
    else:
        tp_rank = rank % tp_size

    # Model path (1.5B model for testing)
    model_path = "/home/sd/npu_train/models/Qwen-Qwen2.5-1.5B-Instruct"

    if rank == 0:
        print(f"\n{'='*60}")
        print("Simple TP Training Test")
        print(f"{'='*60}")
        print(f"Model: {model_path}")
        print(f"TP Size: {tp_size}")
        print(f"World Size: {world_size}")
        print(f"{'='*60}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with FP16 (HCCL doesn't support BF16 AllReduce)
    if rank == 0:
        print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # FP16 for HCCL
        device_map={"": device},
        trust_remote_code=True,
    )

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Expected parameters per TP rank: {total_params // tp_size:,}")

    # Convert to TP
    if tp_size > 1:
        if rank == 0:
            print(f"Converting to TP (tp_size={tp_size})...")
        model = convert_to_tp(model, tp_size=tp_size, rank=tp_rank)
        if rank == 0:
            print(f"TP conversion complete!")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    if rank == 0:
        print("\nStarting training test...")

    # Create a simple batch on each rank (no sampler issues)
    batch_size = 1
    max_length = 128

    # Generate synthetic data per rank
    sample_text = "The quick brown fox jumps over the lazy dog. " * 10
    inputs = tokenizer(
        [sample_text] * batch_size,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )

    # Move to device
    input_ids = inputs["input_ids"].to(device)
    labels = input_ids.clone()

    if rank == 0:
        print(f"Input shape: {input_ids.shape}")
        print(f"Labels shape: {labels.shape}")

    # Training loop
    model.train()
    num_steps = 10
    start_time = time.time()

    for step in range(num_steps):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Sync gradients across TP ranks
        if tp_size > 1:
            sync_gradients_tp(model, tp_size)

        optimizer.step()

        if rank == 0:
            elapsed = time.time() - start_time
            loss_value = loss.item()
            print(f"Step {step+1}/{num_steps} | Loss: {loss_value:.4f} | Time: {elapsed:.2f}s")

    # Final sync
    dist.barrier()

    if rank == 0:
        elapsed = time.time() - start_time
        print(f"\nTraining test completed!")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Average time per step: {elapsed/num_steps:.2f}s")

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
