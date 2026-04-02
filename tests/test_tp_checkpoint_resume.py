#!/usr/bin/env python3
"""CPU-only smoke test for TP rank checkpoint save/load with trainer state."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from npu_parallel.checkpoint_utils import load_tp_rank_checkpoint, save_tp_rank_checkpoint


def main() -> None:
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    x = torch.randn(3, 4)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    original_weight = model.weight.detach().clone()
    original_bias = model.bias.detach().clone()
    original_scheduler_state = scheduler.state_dict()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_tp_rank_checkpoint(
            tmpdir,
            model,
            rank=0,
            tp_size=1,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state={"step": 7, "epoch": 1},
            extra_metadata={"source": "test"},
        )

        trainer_state_path = Path(tmpdir) / 'rank_0' / 'trainer_state.pt'
        trainer_state = torch.load(trainer_state_path, map_location='cpu', weights_only=True)
        assert trainer_state['optimizer_state_dict']['param_groups'][0]['param_names'] == ['weight', 'bias']

        # Scramble optimizer param ordering in the saved file to verify name-based restore.
        scrambled = trainer_state['optimizer_state_dict']
        scrambled['param_groups'][0]['param_names'] = ['bias', 'weight']
        scrambled['param_groups'][0]['params'] = [0, 1]
        weight_state = scrambled['state'][0]
        bias_state = scrambled['state'][1]
        scrambled['state'] = {0: bias_state, 1: weight_state}
        trainer_state['optimizer_state_dict'] = scrambled
        torch.save(trainer_state, trainer_state_path)

        new_model = torch.nn.Linear(4, 2)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
        new_scheduler = torch.optim.lr_scheduler.LambdaLR(new_optimizer, lr_lambda=lambda step: 1.0)

        restored = load_tp_rank_checkpoint(
            tmpdir,
            new_model,
            rank=0,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
        )

        assert torch.allclose(new_model.weight, original_weight)
        assert torch.allclose(new_model.bias, original_bias)
        assert restored["step"] == 7
        assert restored["epoch"] == 1
        assert restored["checkpoint_meta"]["source"] == "test"
        assert restored["checkpoint_meta"]["has_trainer_state"] is True
        adapted_state = new_optimizer.state_dict()['state']
        assert torch.equal(adapted_state[0]['exp_avg'], weight_state['exp_avg'])
        assert torch.equal(adapted_state[1]['exp_avg'], bias_state['exp_avg'])
        assert new_scheduler.state_dict() == original_scheduler_state

    print("TP checkpoint save/load resume smoke test: OK")


if __name__ == "__main__":
    main()
