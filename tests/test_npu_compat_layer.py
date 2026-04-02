#!/usr/bin/env python3
"""CPU-only smoke tests for npu_compat fallback helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from npu_parallel.npu_compat import (
    classify_runtime_error,
    compatibility_report,
    get_compat_policy,
    get_fallback_stats,
    get_perf_counters,
    known_error_signatures,
    recommended_action,
    reset_fallback_stats,
    runtime_info,
    safe_has_any_tokens,
    safe_nonzero,
    safe_softmax,
    safe_topk,
    set_compat_policy,
    supports_op,
)


def main() -> None:
    set_compat_policy("fallback")
    reset_fallback_stats()

    info = runtime_info()
    assert "torch_version" in info
    assert "npu_available" in info
    report = compatibility_report(device_type="cpu")
    assert report["device_type"] == "cpu"
    assert "topk" in report["ops"]
    assert "policy" in report
    assert report["policy"]["mode"] == "fallback"
    assert "fallback_stats" in report
    assert "perf_counters" in report
    assert "error_signatures" in report
    assert "patterns" in report["error_signatures"]
    assert "acl_runtime" in report["error_signatures"]["patterns"]

    tensor = torch.tensor([[1.0, 3.0, 2.0], [0.1, 0.2, 0.3]])
    values, indices = safe_topk(tensor, k=2, dim=-1)
    ref_values, ref_indices = torch.topk(tensor, k=2, dim=-1)
    assert torch.equal(values, ref_values)
    assert torch.equal(indices, ref_indices)

    logits = torch.tensor([[1.0, 2.0, 3.0]])
    softmax = safe_softmax(logits, dim=-1)
    ref_softmax = torch.softmax(logits, dim=-1)
    assert torch.allclose(softmax, ref_softmax)

    assert safe_has_any_tokens(torch.tensor([0.0, 0.0, 0.0])) is False
    assert safe_has_any_tokens(torch.tensor([0.0, 1.0, 0.0])) is True

    mask = torch.tensor([[False, True], [True, False]])
    nonzero = safe_nonzero(mask)
    ref_nonzero = torch.nonzero(mask, as_tuple=False)
    assert torch.equal(nonzero, ref_nonzero)

    assert supports_op("topk", device_type="cpu") is True
    assert supports_op("unknown_op", device_type="cpu") is False
    perf_before = get_perf_counters()
    assert perf_before["total_attempts"] >= 4

    original_topk = torch.topk
    try:
        def broken_topk(*args, **kwargs):
            raise RuntimeError("ACL error code 500002 in topk")

        torch.topk = broken_topk
        fallback_values, fallback_indices = safe_topk(tensor, k=1, dim=-1)
        assert fallback_values.shape == (2, 1)
        assert fallback_indices.shape == (2, 1)
    finally:
        torch.topk = original_topk

    fallback_stats = get_fallback_stats()
    assert "topk" in fallback_stats
    assert fallback_stats["topk"]["count"] >= 1
    assert fallback_stats["topk"]["attempts"] >= 2
    assert fallback_stats["topk"]["primary_failures"] >= 1
    assert fallback_stats["topk"]["fallback_time_ms"] >= 0.0
    assert fallback_stats["topk"]["last_error_class"] == "acl_runtime"
    assert isinstance(fallback_stats["topk"]["recommended_action"], str)
    assert classify_runtime_error(RuntimeError("Operator not implemented")) == "unsupported_op"
    assert classify_runtime_error(RuntimeError("HCCL all_to_all timeout")) == "hccl_runtime"
    assert isinstance(known_error_signatures()["memory"], list)
    assert "fallback" in recommended_action("unsupported_op").lower()
    assert get_compat_policy()["mode"] == "fallback"
    perf_after = get_perf_counters()
    assert perf_after["total_fallback_count"] >= 1
    assert perf_after["total_attempts"] >= perf_before["total_attempts"]

    set_compat_policy("strict")
    reset_fallback_stats()
    original_topk = torch.topk
    try:
        def strict_broken_topk(*args, **kwargs):
            raise RuntimeError("topk not implemented for npu")

        torch.topk = strict_broken_topk
        strict_failed = False
        try:
            safe_topk(tensor, k=1, dim=-1)
        except RuntimeError:
            strict_failed = True
        assert strict_failed is True
        strict_stats = get_fallback_stats()
        assert "topk" in strict_stats
        assert strict_stats["topk"]["count"] == 0
        assert strict_stats["topk"]["primary_failures"] >= 1
    finally:
        torch.topk = original_topk
        set_compat_policy("fallback")
        reset_fallback_stats()

    print("NPU compatibility-layer smoke test: OK")


if __name__ == "__main__":
    main()
