#!/usr/bin/env python3
"""CPU-only smoke test for TPAttention compatibility fallback path."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from npu_parallel.tp_attention import TPAttention


def main() -> None:
    module = TPAttention(
        hidden_size=32,
        num_heads=4,
        tp_size=1,
        rank=0,
        use_rope=False,
        use_sdpa=False,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    q = torch.randn(2, 4, 8, 8)
    k = torch.randn(2, 4, 8, 8)
    v = torch.randn(2, 4, 8, 8)

    original_softmax = torch.softmax
    try:
        def broken_softmax(*args, **kwargs):
            raise RuntimeError("ACL error code 500002 in softmax")

        torch.softmax = broken_softmax
        attn_output, attn_weights = module._manual_attention(q, k, v, attention_mask=None)
    finally:
        torch.softmax = original_softmax

    assert attn_output.shape == q.shape
    assert attn_weights.shape == (2, 4, 8, 8)
    print("TP attention compatibility smoke test: OK")


if __name__ == "__main__":
    main()
