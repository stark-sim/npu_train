#!/usr/bin/env python3
"""
简化的 TP 性能测试 - 单次运行一个 TP size
"""

import os
import time
import argparse
import torch
import torch_npu
import torch.distributed as dist
from transformers import AutoModelForCausalLM

# 导入 TP 实现
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from npu_parallel import convert_to_tp


def setup_distributed():
    """初始化分布式环境"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    os.environ.setdefault("HCCL_INTRA_PCIE_ENABLE", "1")
    os.environ.setdefault("HCCL_INTER_PCIE_ENABLE", "1")

    dist.init_process_group(backend="hccl")
    torch.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")

    return rank, world_size, device


def get_memory_gb(device):
    """获取显存使用 (GB)"""
    return torch.npu.memory_reserved(device) / (1024**3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=20)
    args = parser.parse_args()

    rank, world_size, device = setup_distributed()

    if rank == 0:
        print(f"\n{'='*50}")
        print(f"TP={args.tp_size} 性能测试")
        print(f"{'='*50}")
        print(f"模型: {args.model_path}")
        print(f"batch={args.batch_size}, seq_len={args.seq_len}, steps={args.num_steps}")

    # 清空缓存
    torch.npu.empty_cache()
    if rank == 0:
        mem_before = get_memory_gb(device)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )

    if rank == 0:
        mem_load = get_memory_gb(device)
        params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"\n[加载] 参数: {params:.2f}B, 显存: {mem_load:.2f} GB")

    # TP 转换
    if args.tp_size > 1:
        if rank == 0:
            print(f"\n[TP 转换] tp_size={args.tp_size}")
        t0 = time.time()
        model = convert_to_tp(model, tp_size=args.tp_size, rank=rank)
        torch.npu.synchronize()
        t_convert = time.time() - t0
        if rank == 0:
            print(f"[TP 转换] 耗时: {t_convert:.3f}s")

    # Warmup
    for _ in range(3):
        input_ids = torch.randint(0, 151936, (args.batch_size, args.seq_len), device=device)
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

    # Benchmark
    torch.npu.synchronize()
    t0 = time.time()

    for i in range(args.num_steps):
        input_ids = torch.randint(0, 151936, (args.batch_size, args.seq_len), device=device)
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        if rank == 0 and (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            tps = args.batch_size * args.seq_len * (i + 1) / elapsed
            print(f"  Step {i+1}/{args.num_steps}: {tps:.0f} tok/s")

    torch.npu.synchronize()
    t_total = time.time() - t0
    mem_peak = get_memory_gb(device)

    if rank == 0:
        tps = args.batch_size * args.seq_len * args.num_steps / t_total
        print(f"\n[结果]")
        print(f"  总时间: {t_total:.1f}s")
        print(f"  吞吐: {tps:.0f} tokens/sec")
        print(f"  峰值显存: {mem_peak:.2f} GB")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
