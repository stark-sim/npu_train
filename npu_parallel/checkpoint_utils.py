"""Checkpoint helpers for custom tensor-parallel training."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


CHECKPOINT_FORMAT_VERSION = 1


def _barrier_if_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _inject_optimizer_param_names(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_state_dict: dict[str, Any],
) -> dict[str, Any]:
    name_by_param = {id(param): name for name, param in model.named_parameters()}
    param_groups = []
    for live_group, saved_group in zip(optimizer.param_groups, optimizer_state_dict.get("param_groups", [])):
        group_copy = dict(saved_group)
        group_copy["param_names"] = [name_by_param[id(param)] for param in live_group["params"]]
        param_groups.append(group_copy)
    optimizer_state_dict = dict(optimizer_state_dict)
    optimizer_state_dict["param_groups"] = param_groups
    return optimizer_state_dict


def _clone_optimizer_value(value: Any) -> Any:
    return value.clone() if torch.is_tensor(value) else deepcopy(value)


def _adapt_optimizer_state_dict_by_name(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loaded_state_dict: dict[str, Any],
) -> dict[str, Any]:
    current_state_dict = _inject_optimizer_param_names(model, optimizer, optimizer.state_dict())
    adapted_state = {"state": {}, "param_groups": []}

    loaded_name_to_state: dict[str, dict[str, Any]] = {}
    for loaded_group in loaded_state_dict.get("param_groups", []):
        loaded_names = loaded_group.get("param_names", [])
        loaded_ids = loaded_group.get("params", [])
        for param_id, param_name in zip(loaded_ids, loaded_names):
            if param_id in loaded_state_dict.get("state", {}):
                loaded_name_to_state[param_name] = loaded_state_dict["state"][param_id]

    loaded_group_templates = loaded_state_dict.get("param_groups", [])
    for group_index, current_group in enumerate(current_state_dict.get("param_groups", [])):
        adapted_group = dict(current_group)
        if group_index < len(loaded_group_templates):
            for key, value in loaded_group_templates[group_index].items():
                if key not in {"params", "param_names"}:
                    adapted_group[key] = deepcopy(value)
        adapted_state["param_groups"].append(adapted_group)

        current_names = current_group.get("param_names", [])
        current_ids = current_group.get("params", [])
        for current_id, current_name in zip(current_ids, current_names):
            if current_name in loaded_name_to_state:
                adapted_state["state"][current_id] = {
                    key: _clone_optimizer_value(value)
                    for key, value in loaded_name_to_state[current_name].items()
                }

    return adapted_state


def save_tp_rank_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    rank: int,
    tp_size: int,
    *,
    tokenizer: Any | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    training_state: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Save one rank of a TP checkpoint and shared metadata/tokenizer once."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rank_dir = checkpoint_dir / f"rank_{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), rank_dir / "model.pt")

    trainer_state = {}
    if optimizer is not None:
        trainer_state["optimizer_state_dict"] = _inject_optimizer_param_names(model, optimizer, optimizer.state_dict())
    if scheduler is not None:
        trainer_state["scheduler_state_dict"] = scheduler.state_dict()
    if training_state:
        trainer_state.update(training_state)
    if trainer_state:
        torch.save(trainer_state, rank_dir / "trainer_state.pt")

    metadata = {
        "format": "custom_tp_rank_shards",
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "tp_size": tp_size,
        "has_trainer_state": bool(trainer_state),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    _barrier_if_distributed()
    if rank == 0:
        with (checkpoint_dir / "checkpoint_meta.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(checkpoint_dir)
    _barrier_if_distributed()
    return rank_dir


def load_tp_rank_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    rank: int,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load one rank of a TP checkpoint and optionally restore trainer state."""
    checkpoint_dir = Path(checkpoint_dir)
    rank_dir = checkpoint_dir / f"rank_{rank}"
    model_path = rank_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing rank checkpoint: {model_path}")

    model_state_dict = torch.load(model_path, map_location=map_location, weights_only=True)
    model.load_state_dict(model_state_dict, strict=True)

    restored: dict[str, Any] = {}
    trainer_state_path = rank_dir / "trainer_state.pt"
    if trainer_state_path.exists():
        trainer_state = torch.load(trainer_state_path, map_location="cpu", weights_only=True)
        if optimizer is not None and "optimizer_state_dict" in trainer_state:
            adapted_state = _adapt_optimizer_state_dict_by_name(model, optimizer, trainer_state["optimizer_state_dict"])
            optimizer.load_state_dict(adapted_state)
        if scheduler is not None and "scheduler_state_dict" in trainer_state:
            scheduler.load_state_dict(trainer_state["scheduler_state_dict"])
        restored.update({
            key: value
            for key, value in trainer_state.items()
            if key not in {"optimizer_state_dict", "scheduler_state_dict"}
        })

    meta_path = checkpoint_dir / "checkpoint_meta.json"
    if meta_path.exists():
        restored["checkpoint_meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    return restored


def write_tp_state_dict_shards(
    checkpoint_dir: str | Path,
    shard_state_dicts: dict[int, dict[str, torch.Tensor]],
    *,
    trainer_states: dict[int, dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write rank-sharded state_dicts in the repository TP checkpoint format."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for rank, state_dict in sorted(shard_state_dicts.items()):
        rank_dir = checkpoint_dir / f"rank_{rank}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, rank_dir / "model.pt")
        if trainer_states and rank in trainer_states:
            torch.save(trainer_states[rank], rank_dir / "trainer_state.pt")

    payload = {
        "format": "custom_tp_rank_shards",
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "has_trainer_state": bool(trainer_states),
    }
    if metadata:
        payload.update(metadata)
    with (checkpoint_dir / "checkpoint_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
