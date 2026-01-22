#!/usr/bin/env python3
"""
Benchmark Tensor Parallelism vs Data Parallel on NPU

Compare performance between:
1. DDP (Data Parallel) - baseline
2. TP (Tensor Parallel) - PyTorch native
"""

import os
import time
import argparse
import torch
import torch_npu
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


def setup_distributed():
    """Setup distributed environment"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    dist.init_process_group(backend="hccl")
    torch.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")

    return rank, world_size, device


def benchmark_ddp(model, device, rank, batch_size, seq_len, num_steps=20):
    """Benchmark Data Parallel (DDP)"""
    print(f"[Rank {rank}] Benchmarking DDP...")

    # Setup model for DDP
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], bucket_cap_mb=25
    )

    # Warmup
    for _ in range(3):
        input_ids = torch.randint(0, 151936, (batch_size, seq_len), device=device)
        _ = model(input_ids=input_ids)

    # Actual benchmark
    torch.npu.synchronize()
    start_time = time.time()

    for i in range(num_steps):
        input_ids = torch.randint(0, 151936, (batch_size, seq_len), device=device)
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        if i % 10 == 0 and rank == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = batch_size * seq_len * (i + 1) / elapsed
            print(f"[DDP] Step {i}/{num_steps} | Tokens/sec: {tokens_per_sec:.0f}")

    torch.npu.synchronize()
    total_time = time.time() - start_time

    return {
        "total_time": total_time,
        "tokens_per_sec": batch_size * seq_len * num_steps / total_time,
        "steps_per_sec": num_steps / total_time,
    }


def benchmark_tp(model, device, rank, tp_size, batch_size, seq_len, num_steps=20):
    """Benchmark Tensor Parallel (TP)"""
    print(f"[Rank {rank}] Benchmarking TP with tp_size={tp_size}...")

    # Create device mesh
    try:
        tp_mesh = init_device_mesh("npu", (tp_size,))
    except:
        tp_mesh = init_device_mesh("cuda", (tp_size,))

    # Create TP parallelize plan
    tp_plan = {}
    for name, module in model.named_modules():
        if name.endswith(".self_attn.q_proj") or name.endswith(".self_attn.k_proj") or name.endswith(".self_attn.v_proj"):
            tp_plan[name] = ColwiseParallel()
        elif name.endswith(".self_attn.o_proj"):
            tp_plan[name] = RowwiseParallel()
        elif name.endswith(".mlp.gate_proj") or name.endswith(".mlp.up_proj"):
            tp_plan[name] = ColwiseParallel()
        elif name.endswith(".mlp.down_proj"):
            tp_plan[name] = RowwiseParallel()
        elif name.endswith(".lm_head"):
            tp_plan[name] = ColwiseParallel()

    # Apply TP
    model = parallelize_module(model, tp_mesh, tp_plan)

    # Warmup
    for _ in range(3):
        input_ids = torch.randint(0, 151936, (batch_size, seq_len), device=device)
        _ = model(input_ids=input_ids)

    # Actual benchmark
    torch.npu.synchronize()
    start_time = time.time()

    for i in range(num_steps):
        input_ids = torch.randint(0, 151936, (batch_size, seq_len), device=device)
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        if i % 10 == 0 and rank == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = batch_size * seq_len * (i + 1) / elapsed
            print(f"[TP{tp_size}] Step {i}/{num_steps} | Tokens/sec: {tokens_per_sec:.0f}")

    torch.npu.synchronize()
    total_time = time.time() - start_time

    return {
        "total_time": total_time,
        "tokens_per_sec": batch_size * seq_len * num_steps / total_time,
        "steps_per_sec": num_steps / total_time,
    }


def get_memory_usage(device):
    """Get current NPU memory usage"""
    allocated = torch.npu.memory_allocated(device) / 1e9
    reserved = torch.npu.memory_reserved(device) / 1e9
    return {"allocated_gb": allocated, "reserved_gb": reserved}


def main():
    """Main benchmark function"""
    parser = argparse.ArgumentParser(description="Benchmark TP vs DDP on NPU")
    parser.add_argument("--model_path", type=str,
                        default="/home/sd/npu_train/models/AI-ModelScope/qwen2.5-7b-instruct")
    parser.add_argument("--mode", type=str, default="tp",
                        choices=["ddp", "tp", "both"],
                        help="Benchmark mode")
    parser.add_argument("--tp_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--num_steps", type=int, default=50)
    args = parser.parse_args()

    # Setup
    rank, world_size, device = setup_distributed()

    if rank == 0:
        print("\n" + "="*60)
        print("NPU Tensor Parallelism Benchmark")
        print("="*60)
        print(f"Model: {args.model_path}")
        print(f"Mode: {args.mode}")
        print(f"TP Size: {args.tp_size}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Seq Length: {args.seq_len}")
        print(f"World Size: {world_size}")
        print("="*60 + "\n")

    # Load model (keep on CPU first for fair comparison)
    model_dtype = torch.bfloat16 if torch_npu.is_available() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=model_dtype,
        device_map={"": "cpu"},
    )

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}\n")

    results = {}

    # Benchmark based on mode
    if args.mode in ["ddp", "both"]:
        # Move to device for DDP
        model.to(device)
        torch.npu.synchronize()

        # Get memory before
        mem_before = get_memory_usage(device)

        ddp_results = benchmark_ddp(
            model, device, rank, args.batch_size, args.seq_len, args.num_steps
        )

        # Get memory after
        mem_after = get_memory_usage(device)

        ddp_results.update({
            "peak_memory_gb": mem_after["reserved_gb"],
            "memory_delta_gb": mem_after["reserved_gb"] - mem_before["reserved_gb"],
        })

        results["DDP"] = ddp_results

        if rank == 0:
            print(f"\n[DDP Results]")
            print(f"  Total Time: {ddp_results['total_time']:.2f}s")
            print(f"  Tokens/sec: {ddp_results['tokens_per_sec']:.0f}")
            print(f"  Steps/sec: {ddp_results['steps_per_sec']:.2f}")
            print(f"  Peak Memory: {ddp_results['peak_memory_gb']:.1f} GB")

    if args.mode in ["tp", "both"]:
        # Reload model for TP
        del model
        torch.npu.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=model_dtype,
            device_map={"": "cpu"},
        )

        # Move to device
        model.to(device)
        torch.npu.synchronize()

        # Get memory before
        mem_before = get_memory_usage(device)

        tp_results = benchmark_tp(
            model, device, rank, args.tp_size, args.batch_size, args.seq_len, args.num_steps
        )

        # Get memory after
        mem_after = get_memory_usage(device)

        tp_results.update({
            "peak_memory_gb": mem_after["reserved_gb"],
            "memory_delta_gb": mem_after["reserved_gb"] - mem_before["reserved_gb"],
        })

        results[f"TP{args.tp_size}"] = tp_results

        if rank == 0:
            print(f"\n[TP{args.tp_size} Results]")
            print(f"  Total Time: {tp_results['total_time']:.2f}s")
            print(f"  Tokens/sec: {tp_results['tokens_per_sec']:.0f}")
            print(f"  Steps/sec: {tp_results['steps_per_sec']:.2f}")
            print(f"  Peak Memory: {tp_results['peak_memory_gb']:.1f} GB")

    # Compare if both modes
    if args.mode == "both" and rank == 0:
        print("\n" + "="*60)
        print("Comparison")
        print("="*60)

        speedup = results["TP" + str(args.tp_size)]["tokens_per_sec"] / results["DDP"]["tokens_per_sec"]
        memory_reduction = 1 - results["TP" + str(args.tp_size)]["peak_memory_gb"] / results["DDP"]["peak_memory_gb"]

        print(f"TP{args.tp_size} vs DDP:")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Memory Reduction: {memory_reduction*100:.1f}%")
        print("="*60)

    dist.destroy_process_group()

    if rank == 0:
        print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
