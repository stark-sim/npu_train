#!/usr/bin/env python3
"""
Simple DDP Training Test on NPU
"""

import os
import time
import torch
import torch_npu
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM


def setup_npu():
    """Setup NPU device and distributed"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    dist.init_process_group(backend="hccl")
    torch.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")

    return rank, world_size, device


def main():
    rank, world_size, device = setup_npu()

    model_path = "/home/sd/npu_train/models/Qwen-Qwen2.5-1.5B-Instruct"

    if rank == 0:
        print(f"\n{'='*60}")
        print("Simple DDP Training Test")
        print(f"{'='*60}")
        print(f"World Size: {world_size}")
        print(f"{'='*60}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with FP16
    if rank == 0:
        print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

    # Wrap with DDP
    if rank == 0:
        print("Wrapping with DDP...")
    model = DDP(model, device_ids=[rank])

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
    )

    if rank == 0:
        print("\nStarting training test...")

    # Create simple batch
    batch_size = 1
    seq_len = 64
    sample_text = "The quick brown fox jumps over the lazy dog. " * 10

    inputs = tokenizer(
        [sample_text] * batch_size,
        truncation=True,
        max_length=seq_len,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(device)
    labels = input_ids.clone()

    if rank == 0:
        print(f"Input shape: {input_ids.shape}")

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

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if rank == 0:
            elapsed = time.time() - start_time
            loss_value = loss.item()
            print(f"Step {step+1}/{num_steps} | Loss: {loss_value:.4f} | Time: {elapsed:.2f}s")

    dist.barrier()

    if rank == 0:
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"DDP training test completed!")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Average time per step: {elapsed/num_steps:.2f}s")
        print(f"{'='*60}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
