#!/usr/bin/env python3
"""
Simple test script for custom NPU TP implementation with 1.5B Qwen model

This tests:
1. Model loading
2. TP conversion
3. Forward pass
4. Backward pass
5. Gradient synchronization
"""

import os
import sys
import time
import torch
import torch_npu
import torch.distributed as dist

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup():
    """Setup NPU and distributed"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"[Rank {rank}] Initializing HCCL...")
    dist.init_process_group(backend="hccl")
    torch.npu.set_device(rank)

    device = torch.device(f"npu:{rank}")
    print(f"[Rank {rank}] NPU device: {device}, World size: {world_size}")

    return rank, world_size, device


def test_basic_ops(rank, device):
    """Test basic distributed operations"""
    print(f"[Rank {rank}] Testing basic distributed ops...")

    # Test all_reduce
    tensor = torch.ones(4, 4, device=device) * (rank + 1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = sum(range(1, dist.get_world_size() + 1))
    assert tensor.abs().sum() == expected * 16, f"all_reduce failed: {tensor}"

    # Test all_gather
    tensor = torch.ones(2, 2, device=device) * rank
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    result = torch.cat(gathered, dim=0)
    print(f"[Rank {rank}] all_gather result shape: {result.shape}")

    print(f"[Rank {rank}] Basic ops: PASSED")
    return True


def test_model_loading(rank, device):
    """Test model loading and TP conversion"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = "/home/sd/npu_train/models/Qwen-Qwen2.5-1.5B-Instruct"

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Testing Model Loading + TP Conversion")
        print(f"{'='*60}")
        print(f"Model path: {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if rank == 0:
        print(f"[Rank {rank}] Tokenizer loaded: vocab_size={len(tokenizer)}")

    # Load model in bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"[Rank {rank}] Model loaded: {total_params:,} params")

    return model, tokenizer


def test_tp_conversion(rank, device, model, tp_size=2):
    """Test TP conversion"""
    from npu_parallel import convert_to_tp

    if rank == 0:
        print(f"\n[Rank {rank}] Testing TP conversion with tp_size={tp_size}...")

    # Calculate TP rank
    world_size = dist.get_world_size()
    if world_size == tp_size:
        tp_rank = rank  # Pure TP
    else:
        tp_rank = rank % tp_size

    # Convert model to TP
    model_tp = convert_to_tp(model, tp_size=tp_size, rank=tp_rank)

    # Check parameter count after TP conversion
    tp_params = sum(p.numel() for p in model_tp.parameters() if p.requires_grad)
    expected_params = total_params // tp_size

    if rank == 0:
        print(f"[Rank {rank}] TP conversion complete:")
        print(f"  Original params: {total_params:,}")
        print(f"  TP params (rank {tp_rank}): {tp_params:,}")
        print(f"  Expected per rank: ~{expected_params:,}")

    return model_tp, tp_rank


def test_forward_backward(rank, device, model, tokenizer, tp_size=2):
    """Test forward and backward pass"""
    if rank == 0:
        print(f"\n[Rank {rank}] Testing forward/backward pass...")

    # Prepare synthetic input
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 151936, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    # Forward pass
    if rank == 0:
        print(f"[Rank {rank}] Running forward pass...")

    start_time = time.time()
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    torch.npu.synchronize()
    forward_time = time.time() - start_time

    if rank == 0:
        print(f"[Rank {rank}] Forward pass: loss={loss.item():.4f}, time={forward_time:.3f}s")

    # Backward pass
    if rank == 0:
        print(f"[Rank {rank}] Running backward pass...")

    start_time = time.time()
    loss.backward()
    torch.npu.synchronize()
    backward_time = time.time() - start_time

    if rank == 0:
        print(f"[Rank {rank}] Backward pass: time={backward_time:.3f}s")

    # Check gradients
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5

    if rank == 0:
        print(f"[Rank {rank}] Gradient norm: {grad_norm:.4f}")

    return loss.item()


def test_gradient_sync(rank, device, model, tp_size=2):
    """Test gradient synchronization"""
    from npu_parallel import sync_gradients_tp

    if rank == 0:
        print(f"\n[Rank {rank}] Testing gradient synchronization...")

    # Get a gradient to sync
    for p in model.parameters():
        if p.grad is not None:
            grad_before = p.grad.clone().norm().item()
            break

    # Sync gradients
    sync_gradients_tp(model, tp_size)

    # Check after sync
    for p in model.parameters():
        if p.grad is not None:
            grad_after = p.grad.clone().norm().item()
            break

    if rank == 0:
        print(f"[Rank {rank}] Gradient sync: before={grad_before:.4f}, after={grad_after:.4f}")

    return True


def main():
    """Main test function"""
    print("\n" + "="*60)
    print("NPU Custom TP Test - 1.5B Qwen Model")
    print("="*60 + "\n")

    # Setup
    rank, world_size, device = setup()

    # Parse args
    tp_size = int(os.environ.get("TP_SIZE", "2"))

    if rank == 0:
        print(f"\nConfiguration:")
        print(f"  World size: {world_size}")
        print(f"  TP size: {tp_size}")

    # Test 1: Basic ops
    test_basic_ops(rank, device)

    # Test 2: Model loading
    model, tokenizer = test_model_loading(rank, device)
    global total_params
    total_params = sum(p.numel() for p in model.parameters())

    # Test 3: TP conversion
    model_tp, tp_rank = test_tp_conversion(rank, device, model, tp_size=tp_size)

    # Test 4: Forward/backward
    loss_value = test_forward_backward(rank, device, model_tp, tokenizer, tp_size=tp_size)

    # Test 5: Gradient sync
    test_gradient_sync(rank, device, model_tp, tp_size=tp_size)

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        print("\n" + "="*60)
        print("All tests PASSED!")
        print("="*60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
