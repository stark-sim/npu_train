#!/usr/bin/env python3
"""
Test TP conversion step by step
"""

import os
import torch
import torch_npu
import torch.distributed as dist
from transformers import AutoModelForCausalLM
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

    return rank, world_size, device


def main():
    rank, world_size, device = setup_npu()
    tp_size = 4
    tp_rank = rank  # For pure TP

    model_path = "/home/sd/npu_train/models/Qwen-Qwen2.5-1.5B-Instruct"

    if rank == 0:
        print(f"\n{'='*60}")
        print("Test TP Conversion")
        print(f"{'='*60}")
        print(f"Rank: {rank}, TP Rank: {tp_rank}, TP Size: {tp_size}")

    # Load model
    if rank == 0:
        print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )

    if rank == 0:
        print(f"Model loaded. Params before TP: {sum(p.numel() for p in model.parameters()):,}")

    dist.barrier()

    # Convert to TP
    if rank == 0:
        print("Converting to TP...")

    model = convert_to_tp(model, tp_size=tp_size, rank=tp_rank)

    if rank == 0:
        print(f"TP conversion complete. Params after TP: {sum(p.numel() for p in model.parameters()):,}")

    dist.barrier()

    # Create a simple test input
    batch_size = 1
    seq_len = 32
    hidden_size = 1536  # Qwen 1.5B hidden size

    if rank == 0:
        print(f"\nCreating test input: [{batch_size}, {seq_len}, {hidden_size}]")

    # Create fake hidden states (skip embedding layer)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device=device)

    if rank == 0:
        print("Testing first decoder layer forward pass...")

    # Test the first decoder layer
    layer_0 = model.model.layers[0]

    try:
        with torch.no_grad():
            output = layer_0(hidden_states)[0]

        if rank == 0:
            print(f"Forward pass successful!")
            print(f"Output shape: {output.shape}")
            print(f"Output mean: {output.mean().item():.4f}")
            print(f"Output std: {output.std().item():.4f}")

    except Exception as e:
        if rank == 0:
            print(f"Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        dist.destroy_process_group()
        return

    dist.barrier()

    # Test full model forward pass
    if rank == 0:
        print("\nTesting full model forward pass...")

    # Create token ids input
    input_ids = torch.randint(0, 151936, (batch_size, seq_len), device=device)

    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)

        if rank == 0:
            print(f"Full forward pass successful!")
            print(f"Loss: {outputs.loss.item():.4f}")

    except Exception as e:
        if rank == 0:
            print(f"Full forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()

    dist.barrier()

    if rank == 0:
        print(f"\n{'='*60}")
        print("TP test completed!")
        print(f"{'='*60}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
