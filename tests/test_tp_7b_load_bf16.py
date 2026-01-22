#!/usr/bin/env python3
"""
Test loading 7B model with TP using bfloat16
"""

import os
import sys
import torch
import torch_npu
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM
from npu_parallel import convert_to_tp


def main():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"[Rank {rank}] Initializing HCCL...")
    dist.init_process_group(backend="hccl")
    torch.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")

    print(f"[Rank {rank}] NPU device: {device}, World size: {world_size}")
    tp_size = world_size
    tp_rank = rank

    model_path = "/home/sd/npu_train/models/Qwen-Qwen2.5-7B-Instruct"

    if rank == 0:
        print(f"\nLoading model from {model_path} with bfloat16...")

    # Load model with bfloat16 (half memory)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Expected parameters per TP rank: {total_params // tp_size:,}")

    if rank == 0:
        print(f"\nConverting to TP (tp_size={tp_size})...")

    # Convert to TP
    model = convert_to_tp(model, tp_size=tp_size, rank=tp_rank)

    if rank == 0:
        print(f"TP conversion complete!")

    # Try a simple forward pass
    if rank == 0:
        print(f"\nTesting forward pass...")

    # Create a small input
    input_ids = torch.randint(0, 1000, (1, 8), device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits

    if rank == 0:
        print(f"Forward pass successful!")
        print(f"Input shape: {input_ids.shape}")
        print(f"Logits shape: {logits.shape}")
        print(f"Logits sum: {logits.sum().item():.2f}")

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        print("\nAll tests PASSED!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
