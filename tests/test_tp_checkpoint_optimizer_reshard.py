#!/usr/bin/env python3
"""CPU-only smoke test for TP optimizer-state resharding."""

from __future__ import annotations

from collections import OrderedDict
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.tp_checkpoint import (
    infer_sharding_plan,
    merge_sharded_state_dicts,
    reshard_merged_state_dict,
    reshard_optimizer_states,
)


def build_sample_shards() -> OrderedDict[int, OrderedDict[str, torch.Tensor]]:
    return OrderedDict(
        {
            0: OrderedDict(
                {
                    "linear_col.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                    "linear_col.bias": torch.tensor([0.1, 0.2]),
                    "linear_row.weight": torch.tensor([[10.0, 11.0], [12.0, 13.0]]),
                    "replicated.weight": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
                }
            ),
            1: OrderedDict(
                {
                    "linear_col.weight": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
                    "linear_col.bias": torch.tensor([0.3, 0.4]),
                    "linear_row.weight": torch.tensor([[14.0, 15.0], [16.0, 17.0]]),
                    "linear_row.bias": torch.tensor([0.5, 0.6]),
                    "replicated.weight": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
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
        }
    )


def build_source_trainer_states() -> OrderedDict[int, dict]:
    rank0_names = ["linear_col.weight", "linear_col.bias", "linear_row.weight", "replicated.weight"]
    rank1_names = ["linear_col.weight", "linear_col.bias", "linear_row.weight", "linear_row.bias", "replicated.weight"]

    def build_state(names, tensors):
        state = {}
        params = []
        for idx, (name, tensor) in enumerate(zip(names, tensors)):
            params.append(idx)
            state[idx] = {
                "exp_avg": tensor.clone(),
                "exp_avg_sq": tensor.clone() * 10,
                "step": torch.tensor(5.0),
            }
        return {
            "optimizer_state_dict": {
                "state": state,
                "param_groups": [{
                    "lr": 1e-4,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                    "weight_decay": 0.01,
                    "params": params,
                    "param_names": names,
                }],
            },
            "step": 12,
            "epoch": 0,
        }

    rank0_tensors = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([0.1, 0.2]),
        torch.tensor([[10.0, 11.0], [12.0, 13.0]]),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
    ]
    rank1_tensors = [
        torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        torch.tensor([0.3, 0.4]),
        torch.tensor([[14.0, 15.0], [16.0, 17.0]]),
        torch.tensor([0.5, 0.6]),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
    ]

    return OrderedDict({
        0: build_state(rank0_names, rank0_tensors),
        1: build_state(rank1_names, rank1_tensors),
    })


def main() -> None:
    shards = build_sample_shards()
    target = build_target_template()
    merged = merge_sharded_state_dicts(shards, target)
    plan = infer_sharding_plan(shards, merged)
    target_shards = reshard_merged_state_dict(merged, plan, target_tp_size=4)
    trainer_states = build_source_trainer_states()
    resharded = reshard_optimizer_states(
        trainer_states,
        shards,
        merged,
        plan,
        target_shards,
        target_tp_size=4,
    )

    assert resharded[0]['optimizer_state_dict']['param_groups'][0]['param_names'][0] == 'linear_col.weight'
    rank0_first_id = resharded[0]['optimizer_state_dict']['param_groups'][0]['params'][0]
    assert torch.equal(resharded[0]['optimizer_state_dict']['state'][rank0_first_id]['exp_avg'], torch.tensor([[1.0, 2.0]]))

    rank3_names = resharded[3]['optimizer_state_dict']['param_groups'][0]['param_names']
    assert 'linear_row.bias' in rank3_names
    bias_idx = rank3_names.index('linear_row.bias')
    bias_param_id = resharded[3]['optimizer_state_dict']['param_groups'][0]['params'][bias_idx]
    assert torch.equal(resharded[3]['optimizer_state_dict']['state'][bias_param_id]['exp_avg'], torch.tensor([0.5, 0.6]))
    assert resharded[2]['step'] == 12

    print('TP optimizer-state reshard smoke test: OK')


if __name__ == '__main__':
    main()
