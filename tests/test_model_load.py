#!/usr/bin/env python3
"""
Test model loading and simple forward pass on NPU
"""

import os
import torch
import torch_npu
import torch.distributed as dist
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
        print("Test Model Loading and Forward Pass")
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

    # Create input
    sample_text = "The quick brown fox"
    inputs = tokenizer(
        [sample_text],
        truncation=True,
        max_length=64,
        padding="max_length",
        return_tensors="pt",
    )

    if rank == 0:
        print(f"\nTokenized input shape: {inputs['input_ids'].shape}")
        print(f"Input ids dtype: {inputs['input_ids'].dtype}")

    # Move to device
    if rank == 0:
        print("Moving to device...")

    input_ids = inputs["input_ids"].to(device)
    labels = input_ids.clone()

    if rank == 0:
        print(f"input_ids on device shape: {input_ids.shape}")
        print(f"input_ids device: {input_ids.device}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)

    if rank == 0:
        print(f"\nForward pass successful!")
        print(f"Loss: {outputs.loss.item():.4f}")

    dist.barrier()
    if rank == 0:
        print(f"\n{'='*60}")
        print("All tests passed!")
        print(f"{'='*60}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
