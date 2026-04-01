#!/usr/bin/env python3
"""Inspect, export, and reshard custom TP rank-sharded checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
import re
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from npu_parallel.checkpoint_utils import write_tp_state_dict_shards

RANK_DIR_RE = re.compile(r"rank_(\d+)$")
EXPERT_KEY_RE = re.compile(r"^(.*\.experts\.)(\d+)(\..+)$")


def find_rank_dirs(checkpoint_dir: str | Path) -> list[tuple[int, Path]]:
    checkpoint_dir = Path(checkpoint_dir)
    rank_dirs: list[tuple[int, Path]] = []
    for child in checkpoint_dir.iterdir():
        if not child.is_dir():
            continue
        match = RANK_DIR_RE.match(child.name)
        if match and (child / "model.pt").exists():
            rank_dirs.append((int(match.group(1)), child))
    rank_dirs.sort(key=lambda item: item[0])
    if not rank_dirs:
        raise FileNotFoundError(f"No rank_*/model.pt shards found under {checkpoint_dir}")
    return rank_dirs


def load_rank_state_dicts(checkpoint_dir: str | Path) -> OrderedDict[int, OrderedDict[str, torch.Tensor]]:
    state_dicts: OrderedDict[int, OrderedDict[str, torch.Tensor]] = OrderedDict()
    for rank, rank_dir in find_rank_dirs(checkpoint_dir):
        loaded = torch.load(rank_dir / "model.pt", map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict):
            raise TypeError(f"Shard at {rank_dir} is not a state_dict")
        state_dicts[rank] = OrderedDict((key, value) for key, value in loaded.items())
    return state_dicts


def load_rank_trainer_states(checkpoint_dir: str | Path) -> OrderedDict[int, dict[str, Any]]:
    trainer_states: OrderedDict[int, dict[str, Any]] = OrderedDict()
    for rank, rank_dir in find_rank_dirs(checkpoint_dir):
        trainer_path = rank_dir / "trainer_state.pt"
        if trainer_path.exists():
            trainer_states[rank] = torch.load(trainer_path, map_location="cpu", weights_only=True)
    return trainer_states


def _collect_expert_counts(keys: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in keys:
        match = EXPERT_KEY_RE.match(key)
        if not match:
            continue
        prefix, index_text, _ = match.groups()
        counts[prefix] = max(counts.get(prefix, 0), int(index_text) + 1)
    return counts


def _build_rank_local_expert_counts(
    shard_state_dicts: OrderedDict[int, OrderedDict[str, torch.Tensor]]
) -> dict[str, int]:
    counts_by_prefix: dict[str, int] = {}
    for shard in shard_state_dicts.values():
        for prefix, count in _collect_expert_counts(shard.keys()).items():
            counts_by_prefix[prefix] = max(counts_by_prefix.get(prefix, 0), count)
    return counts_by_prefix


def remap_local_expert_key(
    key: str,
    rank: int,
    local_expert_counts: dict[str, int],
    target_expert_counts: dict[str, int],
) -> str:
    match = EXPERT_KEY_RE.match(key)
    if not match:
        return key

    prefix, local_index_text, suffix = match.groups()
    if prefix not in local_expert_counts or prefix not in target_expert_counts:
        return key

    local_count = local_expert_counts[prefix]
    target_count = target_expert_counts[prefix]
    if target_count <= local_count:
        return key

    local_index = int(local_index_text)
    global_index = rank * local_count + local_index
    if global_index >= target_count:
        return key
    return f"{prefix}{global_index}{suffix}"


def collect_remapped_parts(
    shard_state_dicts: OrderedDict[int, OrderedDict[str, torch.Tensor]],
    target_state_dict: OrderedDict[str, torch.Tensor],
) -> dict[str, list[tuple[int, str, torch.Tensor]]]:
    target_expert_counts = _collect_expert_counts(target_state_dict.keys())
    local_expert_counts = _build_rank_local_expert_counts(shard_state_dicts)

    remapped_parts: dict[str, list[tuple[int, str, torch.Tensor]]] = defaultdict(list)
    for rank, shard in shard_state_dicts.items():
        for key, value in shard.items():
            remapped_key = remap_local_expert_key(
                key,
                rank,
                local_expert_counts=local_expert_counts,
                target_expert_counts=target_expert_counts,
            )
            remapped_parts[remapped_key].append((rank, key, value))
    return remapped_parts


def _parts_match_target_except_dim(parts: list[torch.Tensor], target_shape: tuple[int, ...], dim: int) -> bool:
    if not parts:
        return False
    if any(part.dim() != len(target_shape) for part in parts):
        return False
    for axis in range(len(target_shape)):
        if axis == dim:
            continue
        if any(part.shape[axis] != target_shape[axis] for part in parts):
            return False
    return sum(part.shape[dim] for part in parts) == target_shape[dim]


def merge_tensor_parts(key: str, parts: list[torch.Tensor], target_shape: tuple[int, ...]) -> torch.Tensor:
    if not parts:
        raise KeyError(f"No shard tensors found for key: {key}")
    if len(parts) == 1 and tuple(parts[0].shape) == target_shape:
        return parts[0]

    exact_parts = [part for part in parts if tuple(part.shape) == target_shape]
    if exact_parts:
        return exact_parts[-1]

    if len(target_shape) >= 1 and _parts_match_target_except_dim(parts, target_shape, dim=0):
        return torch.cat(parts, dim=0)
    if len(target_shape) >= 2 and _parts_match_target_except_dim(parts, target_shape, dim=1):
        return torch.cat(parts, dim=1)

    raise ValueError(
        f"Unable to merge key {key}: shard shapes {[tuple(part.shape) for part in parts]} -> target {target_shape}"
    )


def merge_sharded_state_dicts(
    shard_state_dicts: OrderedDict[int, OrderedDict[str, torch.Tensor]],
    target_state_dict: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    remapped_parts = collect_remapped_parts(shard_state_dicts, target_state_dict)
    merged = OrderedDict()
    for key, target_tensor in target_state_dict.items():
        entries = remapped_parts.get(key)
        if not entries:
            raise KeyError(f"Missing merged tensor for target key: {key}")
        merged[key] = merge_tensor_parts(key, [entry[2] for entry in entries], tuple(target_tensor.shape))
    return merged


def infer_sharding_plan(
    shard_state_dicts: OrderedDict[int, OrderedDict[str, torch.Tensor]],
    merged_state_dict: OrderedDict[str, torch.Tensor],
) -> dict[str, dict[str, object]]:
    remapped_parts = collect_remapped_parts(shard_state_dicts, merged_state_dict)
    source_tp_size = len(shard_state_dicts)
    plan: dict[str, dict[str, object]] = {}

    for key, full_tensor in merged_state_dict.items():
        entries = remapped_parts[key]
        parts = [entry[2] for entry in entries]
        ranks = [entry[0] for entry in entries]
        match = EXPERT_KEY_RE.match(key)

        if match:
            prefix, expert_index_text, suffix = match.groups()
            expert_index = int(expert_index_text)
            part = parts[-1]
            if part.dim() >= 1 and part.shape[0] * source_tp_size == full_tensor.shape[0] and part.shape[1:] == full_tensor.shape[1:]:
                inner_kind = "shard_dim0"
            elif part.dim() >= 2 and part.shape[1] * source_tp_size == full_tensor.shape[1] and part.shape[0] == full_tensor.shape[0] and part.shape[2:] == full_tensor.shape[2:]:
                inner_kind = "shard_dim1"
            else:
                inner_kind = "full"
            plan[key] = {
                "kind": "expert",
                "expert_prefix": prefix,
                "expert_suffix": suffix,
                "expert_index": expert_index,
                "inner_kind": inner_kind,
            }
            continue

        if len(entries) == source_tp_size and all(tuple(part.shape) == tuple(full_tensor.shape) for part in parts):
            plan[key] = {"kind": "replicated"}
            continue

        if full_tensor.dim() >= 1 and len(entries) == source_tp_size and _parts_match_target_except_dim(parts, tuple(full_tensor.shape), dim=0):
            plan[key] = {"kind": "shard_dim0"}
            continue

        if full_tensor.dim() >= 2 and len(entries) == source_tp_size and _parts_match_target_except_dim(parts, tuple(full_tensor.shape), dim=1):
            plan[key] = {"kind": "shard_dim1"}
            continue

        if len(entries) == 1 and tuple(parts[0].shape) == tuple(full_tensor.shape):
            owner_rank = ranks[0]
            if owner_rank == source_tp_size - 1:
                plan[key] = {"kind": "last_rank"}
            else:
                plan[key] = {"kind": "single_rank", "owner_rank": owner_rank}
            continue

        raise ValueError(
            f"Unable to infer sharding plan for {key} from shard shapes {[tuple(part.shape) for part in parts]}"
        )

    return plan


def _split_evenly(full_tensor: torch.Tensor, dim: int, parts: int, index: int) -> torch.Tensor:
    if full_tensor.shape[dim] % parts != 0:
        raise ValueError(f"Tensor shape {tuple(full_tensor.shape)} is not divisible by {parts} on dim {dim}")
    chunk_size = full_tensor.shape[dim] // parts
    start = index * chunk_size
    return full_tensor.narrow(dim, start, chunk_size).clone()


def param_owners_for_key(
    key: str,
    spec: dict[str, object],
    *,
    target_tp_size: int,
    expert_counts: dict[str, int],
) -> dict[int, str]:
    kind = spec["kind"]
    if kind == "replicated":
        return {rank: key for rank in range(target_tp_size)}
    if kind in {"shard_dim0", "shard_dim1"}:
        return {rank: key for rank in range(target_tp_size)}
    if kind == "last_rank":
        return {target_tp_size - 1: key}
    if kind == "single_rank":
        owner_rank = int(spec["owner_rank"])
        if owner_rank >= target_tp_size:
            raise ValueError(f"Original owner rank {owner_rank} out of range for target tp_size={target_tp_size}")
        return {owner_rank: key}
    if kind == "expert":
        prefix = str(spec["expert_prefix"])
        suffix = str(spec["expert_suffix"])
        expert_index = int(spec["expert_index"])
        total_experts = expert_counts[prefix]
        if total_experts % target_tp_size != 0:
            raise ValueError(
                f"Expert count {total_experts} for prefix {prefix} is not divisible by target tp_size={target_tp_size}"
            )
        experts_per_rank = total_experts // target_tp_size
        owner_rank = expert_index // experts_per_rank
        local_index = expert_index % experts_per_rank
        return {owner_rank: f"{prefix}{local_index}{suffix}"}
    raise ValueError(f"Unsupported sharding kind: {kind}")


def reshard_merged_state_dict(
    merged_state_dict: OrderedDict[str, torch.Tensor],
    sharding_plan: dict[str, dict[str, object]],
    *,
    target_tp_size: int,
) -> OrderedDict[int, OrderedDict[str, torch.Tensor]]:
    target_shards: OrderedDict[int, OrderedDict[str, torch.Tensor]] = OrderedDict(
        (rank, OrderedDict()) for rank in range(target_tp_size)
    )
    expert_counts = _collect_expert_counts(merged_state_dict.keys())

    for key, tensor in merged_state_dict.items():
        spec = sharding_plan[key]
        kind = spec["kind"]

        if kind == "replicated":
            for rank in range(target_tp_size):
                target_shards[rank][key] = tensor.clone()
            continue

        if kind == "shard_dim0":
            for rank in range(target_tp_size):
                target_shards[rank][key] = _split_evenly(tensor, 0, target_tp_size, rank)
            continue

        if kind == "shard_dim1":
            for rank in range(target_tp_size):
                target_shards[rank][key] = _split_evenly(tensor, 1, target_tp_size, rank)
            continue

        if kind == "last_rank":
            target_shards[target_tp_size - 1][key] = tensor.clone()
            continue

        if kind == "single_rank":
            owner_rank = int(spec["owner_rank"])
            if owner_rank >= target_tp_size:
                raise ValueError(f"Original owner rank {owner_rank} out of range for target tp_size={target_tp_size}")
            target_shards[owner_rank][key] = tensor.clone()
            continue

        if kind == "expert":
            owner_mapping = param_owners_for_key(key, spec, target_tp_size=target_tp_size, expert_counts=expert_counts)
            owner_rank, local_key = next(iter(owner_mapping.items()))
            inner_kind = spec["inner_kind"]
            if inner_kind == "shard_dim0":
                target_shards[owner_rank][local_key] = _split_evenly(tensor, 0, target_tp_size, owner_rank)
            elif inner_kind == "shard_dim1":
                target_shards[owner_rank][local_key] = _split_evenly(tensor, 1, target_tp_size, owner_rank)
            else:
                target_shards[owner_rank][local_key] = tensor.clone()
            continue

        raise ValueError(f"Unsupported sharding kind: {kind}")

    return target_shards


def merge_param_states_by_name(
    source_optimizer_entries: dict[str, dict[str, Any]],
    merged_state_dict: OrderedDict[str, torch.Tensor],
    sharding_plan: dict[str, dict[str, object]],
) -> dict[str, dict[str, Any]]:
    merged_entries: dict[str, dict[str, Any]] = {}
    for name, full_param in merged_state_dict.items():
        spec = sharding_plan[name]
        kind = spec["kind"]
        entry = source_optimizer_entries.get(name)
        if entry is None:
            continue
        param_state = entry["state"]
        merged_state: dict[str, Any] = {}
        for state_key, state_value in param_state.items():
            if torch.is_tensor(state_value) and tuple(state_value.shape) == tuple(full_param.shape):
                merged_state[state_key] = state_value.clone()
                continue
            if torch.is_tensor(state_value) and kind in {"shard_dim0", "shard_dim1"}:
                dim = 0 if kind == "shard_dim0" else 1
                if len(full_param.shape) > dim and state_value.dim() == full_param.dim() and _parts_match_target_except_dim([state_value for _ in range(1)], tuple(state_value.shape), dim):
                    merged_state[state_key] = state_value.clone()
                    continue
            merged_state[state_key] = state_value.clone() if torch.is_tensor(state_value) else deepcopy(state_value)
        merged_entries[name] = {"state": merged_state}
    return merged_entries


def collect_source_optimizer_entries(
    trainer_states: OrderedDict[int, dict[str, Any]],
    shard_state_dicts: OrderedDict[int, OrderedDict[str, torch.Tensor]],
    merged_state_dict: OrderedDict[str, torch.Tensor],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any] | None]:
    target_expert_counts = _collect_expert_counts(merged_state_dict.keys())
    local_expert_counts = _build_rank_local_expert_counts(shard_state_dicts)
    entries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_template: dict[str, Any] | None = None

    for rank, trainer_state in trainer_states.items():
        optimizer_state = trainer_state.get("optimizer_state_dict")
        if not optimizer_state:
            continue
        if len(optimizer_state.get("param_groups", [])) != 1:
            raise ValueError("Optimizer-state resharding currently supports a single optimizer param group.")
        param_group = optimizer_state["param_groups"][0]
        param_names = param_group.get("param_names")
        param_ids = param_group.get("params")
        if not param_names or len(param_names) != len(param_ids):
            raise ValueError("Optimizer state is missing param_names; cannot safely reshard optimizer state.")
        if group_template is None:
            group_template = {k: deepcopy(v) for k, v in param_group.items() if k not in {"params", "param_names"}}

        for param_id, local_name in zip(param_ids, param_names):
            remapped_name = remap_local_expert_key(
                local_name,
                rank,
                local_expert_counts=local_expert_counts,
                target_expert_counts=target_expert_counts,
            )
            state = optimizer_state.get("state", {}).get(param_id, {})
            entries[remapped_name].append({"rank": rank, "state": deepcopy(state)})

    return entries, group_template


def _infer_merge_dim_from_spec(spec: dict[str, object]) -> int | None:
    kind = spec["kind"]
    if kind == "shard_dim0":
        return 0
    if kind == "shard_dim1":
        return 1
    if kind == "expert":
        inner_kind = spec["inner_kind"]
        if inner_kind == "shard_dim0":
            return 0
        if inner_kind == "shard_dim1":
            return 1
    return None


def merge_optimizer_entries_by_name(
    source_optimizer_entries: dict[str, list[dict[str, Any]]],
    merged_state_dict: OrderedDict[str, torch.Tensor],
    sharding_plan: dict[str, dict[str, object]],
) -> dict[str, dict[str, Any]]:
    merged_entries: dict[str, dict[str, Any]] = {}
    for name, entry_list in source_optimizer_entries.items():
        if name not in merged_state_dict:
            continue
        full_param = merged_state_dict[name]
        spec = sharding_plan[name]
        merge_dim = _infer_merge_dim_from_spec(spec)
        state_keys = set()
        for entry in entry_list:
            state_keys.update(entry["state"].keys())

        merged_state: dict[str, Any] = {}
        for state_key in state_keys:
            values = [entry["state"][state_key] for entry in entry_list if state_key in entry["state"]]
            if not values:
                continue
            first = values[0]
            if torch.is_tensor(first):
                exact_values = [value for value in values if tuple(value.shape) == tuple(full_param.shape)]
                if exact_values:
                    merged_state[state_key] = exact_values[-1].clone()
                elif merge_dim is not None and all(torch.is_tensor(value) for value in values) and _parts_match_target_except_dim(values, tuple(full_param.shape), merge_dim):
                    merged_state[state_key] = torch.cat(values, dim=merge_dim)
                else:
                    merged_state[state_key] = first.clone()
            else:
                merged_state[state_key] = deepcopy(first)
        merged_entries[name] = {"state": merged_state}
    return merged_entries


def _reshard_state_value(value: Any, full_param: torch.Tensor, spec: dict[str, object], rank: int, target_tp_size: int) -> Any:
    kind = spec["kind"]
    if not torch.is_tensor(value):
        return deepcopy(value)
    if value.dim() == 0 or tuple(value.shape) != tuple(full_param.shape):
        return value.clone()
    if kind == "shard_dim0":
        return _split_evenly(value, 0, target_tp_size, rank)
    if kind == "shard_dim1":
        return _split_evenly(value, 1, target_tp_size, rank)
    if kind == "expert":
        inner_kind = spec["inner_kind"]
        if inner_kind == "shard_dim0":
            return _split_evenly(value, 0, target_tp_size, rank)
        if inner_kind == "shard_dim1":
            return _split_evenly(value, 1, target_tp_size, rank)
    return value.clone()


def reshard_optimizer_states(
    trainer_states: OrderedDict[int, dict[str, Any]],
    shard_state_dicts: OrderedDict[int, OrderedDict[str, torch.Tensor]],
    merged_state_dict: OrderedDict[str, torch.Tensor],
    sharding_plan: dict[str, dict[str, object]],
    target_shards: OrderedDict[int, OrderedDict[str, torch.Tensor]],
    *,
    target_tp_size: int,
) -> dict[int, dict[str, Any]]:
    source_entry_parts, group_template = collect_source_optimizer_entries(trainer_states, shard_state_dicts, merged_state_dict)
    source_entries = merge_optimizer_entries_by_name(source_entry_parts, merged_state_dict, sharding_plan)
    if group_template is None:
        return {}

    expert_counts = _collect_expert_counts(merged_state_dict.keys())
    target_trainer_states: dict[int, dict[str, Any]] = {
        rank: {
            "optimizer_state_dict": {
                "state": {},
                "param_groups": [dict(group_template, params=[], param_names=[])],
            }
        }
        for rank in target_shards.keys()
    }

    next_param_id = {rank: 0 for rank in target_shards.keys()}
    for full_name, full_param in merged_state_dict.items():
        spec = sharding_plan[full_name]
        owner_mapping = param_owners_for_key(full_name, spec, target_tp_size=target_tp_size, expert_counts=expert_counts)
        source_entry = source_entries.get(full_name)

        for rank, local_name in owner_mapping.items():
            param_id = next_param_id[rank]
            next_param_id[rank] += 1
            opt_state_dict = target_trainer_states[rank]["optimizer_state_dict"]
            opt_state_dict["param_groups"][0]["params"].append(param_id)
            opt_state_dict["param_groups"][0]["param_names"].append(local_name)

            if source_entry is None:
                continue
            target_state = {}
            for state_key, state_value in source_entry["state"].items():
                target_state[state_key] = _reshard_state_value(state_value, full_param, spec, rank, target_tp_size)
            opt_state_dict["state"][param_id] = target_state

    source_meta = next(iter(trainer_states.values())) if trainer_states else {}
    passthrough_keys = ["step", "epoch", "total_aux_loss", "final"]
    passthrough = {key: source_meta[key] for key in passthrough_keys if key in source_meta}
    for rank in target_trainer_states:
        target_trainer_states[rank].update(deepcopy(passthrough))
    return target_trainer_states


def load_checkpoint_metadata(checkpoint_dir: str | Path) -> dict[str, object] | None:
    meta_path = Path(checkpoint_dir) / "checkpoint_meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _copy_tokenizer(output_dir: Path, checkpoint_dir: Path, model_path: str, trust_remote_code: bool) -> None:
    from transformers import AutoTokenizer

    tokenizer_source = checkpoint_dir if (checkpoint_dir / "tokenizer_config.json").exists() else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code)
    tokenizer.save_pretrained(output_dir)


def _load_model_state_template(model_path: str, trust_remote_code: bool) -> OrderedDict[str, torch.Tensor]:
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    return OrderedDict((key, value.detach().cpu()) for key, value in model.state_dict().items())


def cmd_inspect(args: argparse.Namespace) -> int:
    checkpoint_dir = Path(args.checkpoint_dir)
    rank_dirs = find_rank_dirs(checkpoint_dir)
    shard_state_dicts = load_rank_state_dicts(checkpoint_dir)
    rank_sizes = {rank: len(state_dict) for rank, state_dict in shard_state_dicts.items()}
    trainer_state_presence = {
        rank: (rank_dir / "trainer_state.pt").exists()
        for rank, rank_dir in rank_dirs
    }
    metadata = load_checkpoint_metadata(checkpoint_dir)

    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Ranks: {[rank for rank, _ in rank_dirs]}")
    print(f"Tensor counts per rank: {rank_sizes}")
    print(f"Trainer state per rank: {trainer_state_presence}")
    if metadata is not None:
        print("Metadata:")
        print(json.dumps(metadata, indent=2, sort_keys=True))
    sample_rank = next(iter(shard_state_dicts))
    sample_keys = list(shard_state_dicts[sample_rank].keys())[: args.sample_keys]
    if sample_keys:
        print("Sample keys:")
        for key in sample_keys:
            print(f"- {key}: {tuple(shard_state_dicts[sample_rank][key].shape)}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_state_dicts = load_rank_state_dicts(checkpoint_dir)
    target_state_dict = _load_model_state_template(args.model_path, args.trust_remote_code)
    merged_state_dict = merge_sharded_state_dicts(shard_state_dicts, target_state_dict)

    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)
    model.load_state_dict(merged_state_dict, strict=True)
    model.save_pretrained(output_dir, max_shard_size=args.max_shard_size)
    _copy_tokenizer(output_dir, checkpoint_dir, args.model_path, args.trust_remote_code)

    if args.save_state_dict:
        torch.save(merged_state_dict, args.save_state_dict)

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "num_ranks": len(shard_state_dicts),
        "num_tensors": len(merged_state_dict),
        "format": "merged_huggingface_model",
    }
    (output_dir / "tp_merge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_reshard(args: argparse.Namespace) -> int:
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_state_dicts = load_rank_state_dicts(checkpoint_dir)
    trainer_states = load_rank_trainer_states(checkpoint_dir)
    target_state_dict = _load_model_state_template(args.model_path, args.trust_remote_code)
    merged_state_dict = merge_sharded_state_dicts(shard_state_dicts, target_state_dict)
    sharding_plan = infer_sharding_plan(shard_state_dicts, merged_state_dict)
    target_shards = reshard_merged_state_dict(
        merged_state_dict,
        sharding_plan,
        target_tp_size=args.target_tp_size,
    )
    target_trainer_states = {}
    if args.include_optimizer_state and trainer_states:
        target_trainer_states = reshard_optimizer_states(
            trainer_states,
            shard_state_dicts,
            merged_state_dict,
            sharding_plan,
            target_shards,
            target_tp_size=args.target_tp_size,
        )

    metadata = {
        "source_checkpoint": str(checkpoint_dir),
        "source_tp_size": len(shard_state_dicts),
        "target_tp_size": args.target_tp_size,
        "resharded": True,
        "has_trainer_state": bool(target_trainer_states),
    }
    write_tp_state_dict_shards(output_dir, target_shards, trainer_states=target_trainer_states or None, metadata=metadata)
    if args.copy_tokenizer:
        _copy_tokenizer(output_dir, checkpoint_dir, args.model_path, args.trust_remote_code)

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "source_tp_size": len(shard_state_dicts),
        "target_tp_size": args.target_tp_size,
        "num_tensors_rank0": len(target_shards[0]),
        "has_trainer_state": bool(target_trainer_states),
        "format": "resharded_tp_rank_shards",
    }
    (output_dir / "tp_reshard_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and export custom TP checkpoints")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a TP checkpoint directory")
    inspect_parser.add_argument("checkpoint_dir")
    inspect_parser.add_argument("--sample-keys", type=int, default=10)
    inspect_parser.set_defaults(func=cmd_inspect)

    export_parser = subparsers.add_parser("export", help="Merge shards and export a HuggingFace model")
    export_parser.add_argument("checkpoint_dir")
    export_parser.add_argument("--model-path", required=True, help="Base model/config path used to rebuild the full model")
    export_parser.add_argument("--output-dir", required=True)
    export_parser.add_argument("--max-shard-size", default="10GB")
    export_parser.add_argument("--save-state-dict", help="Optional path to save the merged raw state_dict via torch.save")
    export_parser.add_argument("--trust-remote-code", action="store_true")
    export_parser.set_defaults(func=cmd_export)

    reshard_parser = subparsers.add_parser("reshard", help="Reshard a TP checkpoint into a new tp_size")
    reshard_parser.add_argument("checkpoint_dir")
    reshard_parser.add_argument("--model-path", required=True, help="Base model/config path used to rebuild the full model")
    reshard_parser.add_argument("--output-dir", required=True)
    reshard_parser.add_argument("--target-tp-size", type=int, required=True)
    reshard_parser.add_argument("--copy-tokenizer", action="store_true")
    reshard_parser.add_argument("--include-optimizer-state", action="store_true")
    reshard_parser.add_argument("--trust-remote-code", action="store_true")
    reshard_parser.set_defaults(func=cmd_reshard)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
