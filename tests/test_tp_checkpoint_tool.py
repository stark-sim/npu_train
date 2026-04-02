#!/usr/bin/env python3
"""CPU-only smoke tests for TP checkpoint merge and reshard logic."""

from __future__ import annotations

from collections import OrderedDict
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.tp_checkpoint import infer_sharding_plan, merge_sharded_state_dicts, reshard_merged_state_dict


def build_sample_shards() -> OrderedDict[int, OrderedDict[str, torch.Tensor]]:
    return OrderedDict(
        {
            0: OrderedDict(
                {
                    "linear_col.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                    "linear_col.bias": torch.tensor([0.1, 0.2]),
                    "linear_row.weight": torch.tensor([[10.0, 11.0], [12.0, 13.0]]),
                    "replicated.weight": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
                    "block.experts.0.gate_proj.weight": torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
                    "block.experts.1.gate_proj.weight": torch.tensor([[3.0, 3.0], [4.0, 4.0]]),
                }
            ),
            1: OrderedDict(
                {
                    "linear_col.weight": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
                    "linear_col.bias": torch.tensor([0.3, 0.4]),
                    "linear_row.weight": torch.tensor([[14.0, 15.0], [16.0, 17.0]]),
                    "linear_row.bias": torch.tensor([0.5, 0.6]),
                    "replicated.weight": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
                    "block.experts.0.gate_proj.weight": torch.tensor([[5.0, 5.0], [6.0, 6.0]]),
                    "block.experts.1.gate_proj.weight": torch.tensor([[7.0, 7.0], [8.0, 8.0]]),
                }
            ),
        }
    )


def build_target_template() -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        {
            "linear_col.weight": torch.empty(4, 2),
            "linear_col.bias": torch.empty(4),
            "linear_row.weight": torch.empty(2, 4),
            "linear_row.bias": torch.empty(2),
            "replicated.weight": torch.empty(2, 2),
            "block.experts.0.gate_proj.weight": torch.empty(2, 2),
            "block.experts.1.gate_proj.weight": torch.empty(2, 2),
            "block.experts.2.gate_proj.weight": torch.empty(2, 2),
            "block.experts.3.gate_proj.weight": torch.empty(2, 2),
        }
    )


def main() -> None:
    shard_state_dicts = build_sample_shards()
    target_state_dict = build_target_template()
    merged = merge_sharded_state_dicts(shard_state_dicts, target_state_dict)

    assert tuple(merged["linear_col.weight"].shape) == (4, 2)
    assert torch.equal(
        merged["linear_col.weight"],
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
    )
    assert torch.equal(merged["linear_col.bias"], torch.tensor([0.1, 0.2, 0.3, 0.4]))
    assert torch.equal(
        merged["linear_row.weight"],
        torch.tensor([[10.0, 11.0, 14.0, 15.0], [12.0, 13.0, 16.0, 17.0]]),
    )
    assert torch.equal(merged["linear_row.bias"], torch.tensor([0.5, 0.6]))
    assert torch.equal(merged["replicated.weight"], torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
    assert torch.equal(merged["block.experts.2.gate_proj.weight"], torch.tensor([[5.0, 5.0], [6.0, 6.0]]))
    assert torch.equal(merged["block.experts.3.gate_proj.weight"], torch.tensor([[7.0, 7.0], [8.0, 8.0]]))

    sharding_plan = infer_sharding_plan(shard_state_dicts, merged)
    resharded = reshard_merged_state_dict(merged, sharding_plan, target_tp_size=4)

    assert list(resharded.keys()) == [0, 1, 2, 3]
    assert torch.equal(resharded[0]["linear_col.weight"], torch.tensor([[1.0, 2.0]]))
    assert torch.equal(resharded[3]["linear_col.bias"], torch.tensor([0.4]))
    assert torch.equal(resharded[0]["linear_row.weight"], torch.tensor([[10.0], [12.0]]))
    assert torch.equal(resharded[3]["linear_row.weight"], torch.tensor([[15.0], [17.0]]))
    assert torch.equal(resharded[3]["linear_row.bias"], torch.tensor([0.5, 0.6]))
    assert "linear_row.bias" not in resharded[0]
    assert torch.equal(resharded[2]["replicated.weight"], torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
    assert torch.equal(resharded[2]["block.experts.0.gate_proj.weight"], torch.tensor([[5.0, 5.0], [6.0, 6.0]]))
    assert torch.equal(resharded[3]["block.experts.0.gate_proj.weight"], torch.tensor([[7.0, 7.0], [8.0, 8.0]]))

    print("TP checkpoint merge + reshard smoke test: OK")


if __name__ == "__main__":
    main()
