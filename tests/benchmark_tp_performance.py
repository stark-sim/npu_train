#!/usr/bin/env python3
"""
性能基准测试：NPU TP 实现对比

测试维度:
1. 内存占用 - 单卡 vs TP
2. 训练吞吐量 - tokens/sec
3. 不同 TP size 的扩展性
"""

import os
import time
import argparse
import torch
import torch_npu
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

# 导入我们的 TP 实现
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from npu_parallel import convert_to_tp, sync_gradients_tp


def setup_distributed():
    """初始化分布式环境"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # HCCL 环境变量
    os.environ.setdefault("HCCL_INTRA_PCIE_ENABLE", "1")
    os.environ.setdefault("HCCL_INTER_PCIE_ENABLE", "1")

    dist.init_process_group(backend="hccl")
    torch.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")

    return rank, world_size, device


def get_memory_info(device):
    """获取 NPU 内存使用情况"""
    allocated = torch.npu.memory_allocated(device) / (1024**3)
    reserved = torch.npu.memory_reserved(device) / (1024**3)
    return {"allocated_gb": allocated, "reserved_gb": reserved}


def benchmark_model_loading(model_path, device, rank, dtype=torch.bfloat16):
    """测试模型加载内存"""
    if rank == 0:
        print(f"\n[模型加载] {model_path}")

    torch.npu.empty_cache()
    mem_before = get_memory_info(device)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
    )

    torch.npu.synchronize()
    mem_after = get_memory_info(device)

    total_params = sum(p.numel() for p in model.parameters())
    memory_used = mem_after["reserved_gb"] - mem_before["reserved_gb"]

    if rank == 0:
        print(f"  参数量: {total_params / 1e9:.2f}B")
        print(f"  显存占用: {memory_used:.2f} GB")

    return model, {
        "total_params": total_params,
        "memory_gb": memory_used,
    }


def benchmark_forward_backward(model, device, rank, batch_size, seq_len, vocab_size, num_steps=20, warmup=3):
    """测试前向+后向性能"""
    if rank == 0:
        print(f"\n[训练性能] batch_size={batch_size}, seq_len={seq_len}")

    # Warmup
    for _ in range(warmup):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

    torch.npu.synchronize()
    mem_before = get_memory_info(device)
    start_time = time.time()

    for i in range(num_steps):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        if rank == 0 and (i + 1) % 5 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = batch_size * seq_len * (i + 1) / elapsed
            print(f"  Step {i+1}/{num_steps}: {tokens_per_sec:.0f} tokens/sec")

    torch.npu.synchronize()
    total_time = time.time() - start_time
    mem_after = get_memory_info(device)

    total_tokens = batch_size * seq_len * num_steps
    avg_tokens_per_sec = total_tokens / total_time
    peak_memory = mem_after["reserved_gb"]

    if rank == 0:
        print(f"\n  结果:")
        print(f"    总时间: {total_time:.2f}s")
        print(f"    平均吞吐: {avg_tokens_per_sec:.0f} tokens/sec")
        print(f"    峰值显存: {peak_memory:.2f} GB")

    return {
        "total_time": total_time,
        "tokens_per_sec": avg_tokens_per_sec,
        "peak_memory_gb": peak_memory,
        "steps_per_sec": num_steps / total_time,
    }


def benchmark_tp(model_path, device, rank, world_size, tp_size, batch_size, seq_len, vocab_size, num_steps=20):
    """测试 TP 性能"""
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"[TP{tp_size} 测试]")
        print(f"{'='*60}")

    torch.npu.empty_cache()

    # 加载模型
    model, load_info = benchmark_model_loading(model_path, device, rank)

    # 转换为 TP
    if rank == 0:
        print(f"\n[TP 转换] tp_size={tp_size}")

    convert_start = time.time()
    model = convert_to_tp(model, tp_size=tp_size, rank=rank)
    torch.npu.synchronize()
    convert_time = time.time() - convert_start

    if rank == 0:
        print(f"  转换耗时: {convert_time:.3f}s")

    # 计算每卡参数量
    params_per_rank = load_info["total_params"] // tp_size
    if rank == 0:
        print(f"  每卡参数: {params_per_rank / 1e9:.2f}B")

    # 测试训练性能
    train_info = benchmark_forward_backward(
        model, device, rank, batch_size, seq_len, vocab_size, num_steps
    )

    # 清理
    del model
    torch.npu.empty_cache()

    return {
        "tp_size": tp_size,
        "load_info": load_info,
        "convert_time": convert_time,
        "train_info": train_info,
    }


def print_comparison(results):
    """打印性能对比"""
    if not results:
        return

    print(f"\n{'='*60}")
    print("性能对比汇总")
    print(f"{'='*60}")

    tp_sizes = sorted([r["tp_size"] for r in results])

    print(f"\n{'TP Size':<10} {'显存/卡(GB)':<15} {'吞吐(tok/s)':<15} {'扩展效率':<10}")
    print("-" * 60)

    baseline_tokens = None
    for r in results:
        tp = r["tp_size"]
        mem = r["train_info"]["peak_memory_gb"]
        tps = r["train_info"]["tokens_per_sec"]

        if tp == 1:
            baseline_tokens = tps
            efficiency = "1.00x"
        else:
            efficiency = f"{tps / baseline_tokens:.2f}x" if baseline_tokens else "N/A"

        print(f"{tp:<10} {mem:<15.2f} {tps:<15.0f} {efficiency:<10}")

    print(f"{'='*60}")

    # 显存节省
    if len(results) >= 2:
        tp1_mem = results[0]["train_info"]["peak_memory_gb"]
        tp_max = max(results, key=lambda x: x["tp_size"])
        tp_max_mem = tp_max["train_info"]["peak_memory_gb"]
        savings = (1 - tp_max_mem / tp1_mem) * 100
        print(f"显存节省: {savings:.1f}% (TP{tp_max['tp_size']} vs TP1)")


def main():
    parser = argparse.ArgumentParser(description="NPU TP 性能基准测试")
    parser.add_argument("--model_path", type=str,
                        default="/home/sd/npu_train/models/AI-ModelScope/qwen2.5-1.5b-instruct")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--vocab_size", type=int, default=151936)
    parser.add_argument("--tp_sizes", type=str, default="1,2,4,8",
                        help="逗号分隔的 TP size 列表")
    args = parser.parse_args()

    rank, world_size, device = setup_distributed()

    if rank == 0:
        print("\n" + "="*60)
        print("NPU Tensor Parallelism 性能基准测试")
        print("="*60)
        print(f"模型: {args.model_path}")
        print(f"配置: batch_size={args.batch_size}, seq_len={args.seq_len}")
        print(f"World Size: {world_size}")
        print("="*60)

    # 解析 TP sizes
    tp_sizes = [int(x) for x in args.tp_sizes.split(",")]
    # 过滤掉超过 world_size 的
    tp_sizes = [tp for tp in tp_sizes if tp <= world_size]

    if not tp_sizes:
        print("无效的 TP size 配置")
        return

    results = []

    # TP=1 作为 baseline (不转换，直接用原模型)
    if 1 in tp_sizes:
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"[Baseline - 无 TP]")
            print(f"{'='*60}")

        torch.npu.empty_cache()
        model, load_info = benchmark_model_loading(args.model_path, device, rank)
        train_info = benchmark_forward_backward(
            model, device, rank, args.batch_size, args.seq_len,
            args.vocab_size, args.num_steps
        )
        del model
        torch.npu.empty_cache()

        results.append({
            "tp_size": 1,
            "load_info": load_info,
            "convert_time": 0,
            "train_info": train_info,
        })

    # 测试其他 TP sizes
    for tp_size in tp_sizes:
        if tp_size == 1:
            continue
        result = benchmark_tp(
            args.model_path, device, rank, world_size, tp_size,
            args.batch_size, args.seq_len, args.vocab_size, args.num_steps
        )
        results.append(result)

    # 打印对比
    if rank == 0:
        print_comparison(results)

    dist.destroy_process_group()

    if rank == 0:
        print("\n测试完成!")


if __name__ == "__main__":
    main()
