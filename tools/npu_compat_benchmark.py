#!/usr/bin/env python3
"""Benchmark npu_compat overhead on CPU or NPU."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from npu_parallel.npu_compat import compatibility_report, reset_fallback_stats, safe_softmax, safe_topk
from npu_parallel.tp_attention import TPAttention
from npu_parallel.tp_moe import TPMoERouter


@contextmanager
def force_torch_failure(symbol_name: str, message: str):
    original = getattr(torch, symbol_name)

    def broken(*args, **kwargs):
        raise RuntimeError(message)

    setattr(torch, symbol_name, broken)
    try:
        yield
    finally:
        setattr(torch, symbol_name, original)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize(device)


def benchmark_case(
    name: str,
    fn: Callable[[], Any],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(max(warmup, 0)):
        fn()
    synchronize_device(device)

    start = time.perf_counter()
    for _ in range(max(iterations, 1)):
        fn()
    synchronize_device(device)
    elapsed = time.perf_counter() - start
    avg_ms = elapsed * 1000.0 / max(iterations, 1)
    return {
        "name": name,
        "iterations": int(max(iterations, 1)),
        "total_ms": round(elapsed * 1000.0, 3),
        "avg_ms": round(avg_ms, 3),
    }


def resolve_device(device_type: str) -> torch.device:
    if device_type == "npu":
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("NPU requested but torch.npu is not available")
        index = torch.npu.current_device()
        torch.npu.set_device(index)
        return torch.device(f"npu:{index}")
    return torch.device("cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark compatibility-layer overhead")
    parser.add_argument("--device-type", choices=["cpu", "npu"], default="cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    reset_fallback_stats()
    device = resolve_device(args.device_type)

    logits = torch.randn(args.batch_size, args.seq_len, args.num_experts, device=device, dtype=torch.float32)
    topk_input = torch.randn(args.batch_size, args.seq_len, args.hidden_size, device=device, dtype=torch.float32)
    q = torch.randn(args.batch_size, args.num_heads, args.seq_len, args.hidden_size // args.num_heads, device=device, dtype=torch.float32)
    k = torch.randn(args.batch_size, args.num_heads, args.seq_len, args.hidden_size // args.num_heads, device=device, dtype=torch.float32)
    v = torch.randn(args.batch_size, args.num_heads, args.seq_len, args.hidden_size // args.num_heads, device=device, dtype=torch.float32)
    router_input = torch.randn(args.batch_size, args.seq_len, args.hidden_size, device=device, dtype=torch.float32)

    attention = TPAttention(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        tp_size=1,
        rank=0,
        use_rope=False,
        use_sdpa=False,
        dtype=torch.float32,
        device=device,
    ).to(device)
    router = TPMoERouter(
        hidden_size=args.hidden_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        dtype=torch.float32,
        device=device,
    ).to(device)

    results = []
    results.append(benchmark_case("softmax_raw", lambda: torch.softmax(logits, dim=-1), device, args.warmup, args.iterations))
    results.append(benchmark_case("softmax_safe", lambda: safe_softmax(logits, dim=-1), device, args.warmup, args.iterations))
    with force_torch_failure("softmax", "ACL error code 500002 in softmax"):
        results.append(
            benchmark_case("softmax_safe_forced_fallback", lambda: safe_softmax(logits, dim=-1), device, args.warmup, args.iterations)
        )

    results.append(benchmark_case("topk_raw", lambda: torch.topk(topk_input, k=args.top_k, dim=-1), device, args.warmup, args.iterations))
    results.append(benchmark_case("topk_safe", lambda: safe_topk(topk_input, k=args.top_k, dim=-1), device, args.warmup, args.iterations))
    with force_torch_failure("topk", "ACL error code 500002 in topk"):
        results.append(
            benchmark_case("topk_safe_forced_fallback", lambda: safe_topk(topk_input, k=args.top_k, dim=-1), device, args.warmup, args.iterations)
        )

    results.append(
        benchmark_case(
            "tp_attention_manual_safe",
            lambda: attention._manual_attention(q, k, v, attention_mask=None),
            device,
            args.warmup,
            args.iterations,
        )
    )
    results.append(
        benchmark_case(
            "tp_moe_router_safe",
            lambda: router(router_input),
            device,
            args.warmup,
            args.iterations,
        )
    )

    summary = {
        "device": str(device),
        "config": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "hidden_size": args.hidden_size,
            "num_heads": args.num_heads,
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "seed": args.seed,
        },
        "compatibility_report": compatibility_report(device_type=args.device_type),
        "results": results,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
