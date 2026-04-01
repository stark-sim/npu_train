"""Runtime checks, policy controls, and safe fallbacks for NPU-sensitive operations."""

from __future__ import annotations

import os
import re
import time
import warnings
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F


ProbeFn = Callable[[torch.device], None]
_POLICY_MODES = {"fallback", "warn", "strict"}
_DEFAULT_POLICY = "fallback"
_OP_RUNTIME_STATS: dict[str, dict[str, Any]] = {}
_WARNED_FALLBACKS: set[tuple[str, str, str]] = set()
_ERROR_SIGNATURES: dict[str, tuple[str, ...]] = {
    "acl_runtime": ("500002", "aclop", "acl", "aicore", "err99999", "ei9999", "ge error", "inner error", "engine_place.cc"),
    "hccl_runtime": ("hccl", "hcom", "allreduce", "all_to_all", "ranktable"),
    "unsupported_op": ("not implemented", "unsupported", "does not support", "not support", "reduceany"),
    "memory": ("out of memory", "oom", "memory allocation", "alloc failed", "cannot allocate memory", "unable to mmap"),
    "timeout": ("timeout", "timed out", "watchdog"),
    "shape_or_dtype": ("shape", "dimension", "size mismatch", "dtype", "scalar type", "storage_offset", "untrustworthy"),
}
_ERROR_RECOMMENDATIONS: dict[str, str] = {
    "acl_runtime": "Enable fallback mode, reduce operator shape complexity, and verify CANN op support matrix.",
    "hccl_runtime": "Check rank topology and timeout settings, then isolate communication path with minimal repro.",
    "unsupported_op": "Keep fallback for continuity and replace unsupported op with equivalent supported kernel.",
    "memory": "Reduce batch/sequence length, enable checkpointing, and inspect allocator split-size settings.",
    "timeout": "Increase HCCL timeout and pre-run warmup to amortize first-step compile latency.",
    "shape_or_dtype": "Validate tensor shape/dtype contracts before op invocation on NPU.",
    "unknown": "Capture full traceback and add a new signature mapping entry with mitigation guidance.",
}
_LOG_ERROR_HINTS = (
    "error",
    "exception",
    "runtimeerror",
    "traceback",
    "failed",
    "failure",
    "crash",
    "fatal",
    "abort",
    "acl",
    "hccl",
    "oom",
)
_IGNORED_WARNING_PATTERNS = (
    "storage_offset",
    "untrustworthy",
    "the oprator of ne is executed",
    "high accuracy but low performance op with 64-bit",
)
_TOKEN_STOPWORDS = {
    "error",
    "runtimeerror",
    "exception",
    "failed",
    "failure",
    "traceback",
    "python",
    "torch",
    "npu",
    "cann",
    "hccl",
    "rank",
    "device",
    "kernel",
    "operator",
    "file",
    "line",
    "most",
    "call",
    "code",
    "func",
    "compiler",
    "depend",
    "terror",
    "stream",
    "model",
    "recent",
    "last",
    "warning",
    "with",
    "result",
    "qwen",
    "qwen2",
    "instruct",
    "modelscope",
    "safetensors",
    "log_inner",
}
_SIGNATURE_CLASS_HINTS: dict[str, tuple[str, ...]] = {
    "acl_runtime": ("acl", "aclop", "aicore", "ge_", "ascend"),
    "hccl_runtime": ("hccl", "hcom", "allreduce", "all_to_all", "ranktable", "collective"),
    "memory": ("oom", "memory", "alloc", "malloc", "workspace"),
    "timeout": ("timeout", "watchdog", "stuck", "hang"),
    "shape_or_dtype": ("shape", "dim", "dimension", "dtype", "broadcast", "stride"),
    "unsupported_op": ("unsupported", "not_impl", "notimplemented", "kernel", "operator", "reduceany"),
}


def _normalize_policy_mode(mode: str | None) -> str:
    if mode is None:
        return _DEFAULT_POLICY
    normalized = str(mode).strip().lower()
    if normalized in _POLICY_MODES:
        return normalized
    return _DEFAULT_POLICY


_COMPAT_POLICY_MODE = _normalize_policy_mode(os.environ.get("NPU_COMPAT_POLICY", _DEFAULT_POLICY))


def _resolve_device(device: torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device(f"npu:{torch.npu.current_device()}")
    return torch.device("cpu")


def _to_python_scalar(value: torch.Tensor) -> float:
    tensor = value.detach()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    return float(tensor.item())


def known_error_signatures() -> dict[str, list[str]]:
    return {name: list(patterns) for name, patterns in _ERROR_SIGNATURES.items()}


def classify_runtime_error(exc: Exception) -> str:
    msg = str(exc).lower()
    for error_class, patterns in _ERROR_SIGNATURES.items():
        if any(token in msg for token in patterns):
            return error_class
    return "unknown"


def recommended_action(error_class: str) -> str:
    return _ERROR_RECOMMENDATIONS.get(error_class, _ERROR_RECOMMENDATIONS["unknown"])


def _extract_unknown_tokens(messages: list[str], top_k: int = 12) -> list[dict[str, Any]]:
    token_counter: Counter[str] = Counter()
    for msg in messages:
        for token in re.findall(r"[a-z0-9_]{4,}", msg.lower()):
            if token in _TOKEN_STOPWORDS:
                continue
            if token.isdigit():
                continue
            token_counter[token] += 1
    suggestions: list[dict[str, Any]] = []
    for token, count in token_counter.most_common(max(top_k, 0)):
        suggestions.append({"candidate_pattern": token, "count": int(count)})
    return suggestions


def analyze_error_messages(
    messages: list[str],
    top_k: int = 12,
    min_count: int = 2,
    max_updates: int = 12,
) -> dict[str, Any]:
    class_counts: Counter[str] = Counter()
    unknown_messages: list[str] = []
    normalized_messages = [str(msg).strip() for msg in messages if str(msg).strip()]
    for msg in normalized_messages:
        error_class = classify_runtime_error(RuntimeError(msg))
        class_counts[error_class] += 1
        if error_class == "unknown":
            unknown_messages.append(msg)
    unknown_suggestions = _extract_unknown_tokens(unknown_messages, top_k=top_k)
    signature_update_plan = build_signature_update_plan(
        class_counts,
        unknown_suggestions,
        min_count=min_count,
        max_updates=max_updates,
    )
    outcome_objective = build_outcome_objective(class_counts, int(class_counts.get("unknown", 0)))
    return {
        "total_messages": len(normalized_messages),
        "class_counts": dict(class_counts),
        "unknown_count": int(class_counts.get("unknown", 0)),
        "unknown_examples": unknown_messages[:5],
        "unknown_signature_suggestions": unknown_suggestions,
        "recommendations": {
            name: recommended_action(name)
            for name, count in class_counts.items()
            if count > 0
        },
        "outcome_objective": outcome_objective,
        "signature_update_plan": signature_update_plan,
        "signature_patch_template": render_signature_patch_template(signature_update_plan),
    }


def _guess_signature_class(candidate_pattern: str) -> str:
    token = candidate_pattern.lower()
    for class_name, hints in _SIGNATURE_CLASS_HINTS.items():
        if any(hint in token for hint in hints):
            return class_name
    return "unknown"


def build_outcome_objective(class_counts: Mapping[str, int], unknown_count: int) -> dict[str, Any]:
    total_errors = int(sum(int(v) for v in class_counts.values()))
    return {
        "goal": "Reduce unknown 910A/CANN runtime failures by converting recurring log patterns into explicit compatibility signatures.",
        "why_it_matters": "Higher signature coverage shortens triage time and improves fallback/action selection consistency.",
        "baseline": {
            "total_classified_error_lines": total_errors,
            "unknown_error_lines": int(unknown_count),
        },
        "success_criteria": [
            "Unknown error ratio trends down across subsequent log scans.",
            "New recurring patterns are mapped to stable classes with explicit recommended actions.",
            "Signature updates stay reviewable and do not auto-modify runtime policy silently.",
        ],
    }


def build_signature_update_plan(
    class_counts: Mapping[str, int],
    unknown_signature_suggestions: list[dict[str, Any]],
    min_count: int = 2,
    max_updates: int = 12,
) -> dict[str, Any]:
    filtered = [
        item
        for item in unknown_signature_suggestions
        if int(item.get("count", 0)) >= max(1, int(min_count))
    ]
    filtered.sort(key=lambda item: int(item.get("count", 0)), reverse=True)
    selected = filtered[: max(0, int(max_updates))]

    proposed_updates: list[dict[str, Any]] = []
    grouped_additions: dict[str, list[str]] = {}
    review_required: list[str] = []
    for item in selected:
        token = str(item.get("candidate_pattern", "")).strip().lower()
        if not token:
            continue
        count = int(item.get("count", 0))
        suggested_class = _guess_signature_class(token)
        confidence = "medium" if suggested_class != "unknown" else "low"
        proposed_updates.append(
            {
                "candidate_pattern": token,
                "count": count,
                "suggested_class": suggested_class,
                "confidence": confidence,
                "recommended_action": recommended_action(suggested_class),
            }
        )
        if suggested_class == "unknown":
            review_required.append(token)
            continue
        grouped_additions.setdefault(suggested_class, []).append(token)

    return {
        "objective": build_outcome_objective(class_counts, int(class_counts.get("unknown", 0))),
        "policy": {
            "min_count_threshold": int(max(1, int(min_count))),
            "max_updates": int(max(0, int(max_updates))),
            "auto_apply": False,
        },
        "proposed_updates": proposed_updates,
        "proposed_signature_additions": grouped_additions,
        "manual_review_required": review_required,
        "target": {
            "file": "npu_parallel/npu_compat.py",
            "variable": "_ERROR_SIGNATURES",
            "workflow": "Review candidates, then add selected tokens to the mapped class tuple.",
        },
    }


def _format_signature_line(class_name: str, patterns: list[str]) -> str:
    quoted = ", ".join(f'"{pattern}"' for pattern in patterns)
    return f'    "{class_name}": ({quoted}),'


def render_signature_patch_template(signature_update_plan: Mapping[str, Any]) -> str:
    additions = signature_update_plan.get("proposed_signature_additions", {})
    if not isinstance(additions, Mapping):
        return ""

    patch_lines = ["*** Begin Patch", "*** Update File: npu_parallel/npu_compat.py"]
    has_changes = False
    for class_name in sorted(additions.keys()):
        if class_name not in _ERROR_SIGNATURES:
            continue
        current_patterns = list(_ERROR_SIGNATURES[class_name])
        candidate_patterns = []
        for token in additions[class_name]:
            normalized = str(token).strip().lower()
            if normalized and normalized not in current_patterns and normalized not in candidate_patterns:
                candidate_patterns.append(normalized)
        if not candidate_patterns:
            continue
        new_patterns = current_patterns + candidate_patterns
        patch_lines.extend(
            [
                "@@",
                f'-{_format_signature_line(class_name, current_patterns)}',
                f'+{_format_signature_line(class_name, new_patterns)}',
            ]
        )
        has_changes = True

    if not has_changes:
        return ""

    patch_lines.append("*** End Patch")
    return "\n".join(patch_lines)


def _should_ignore_log_line(line: str) -> bool:
    normalized = line.lower()
    return any(pattern in normalized for pattern in _IGNORED_WARNING_PATTERNS)


def _extract_error_lines_from_text(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines()]
    return [
        line
        for line in lines
        if line
        and any(hint in line.lower() for hint in _LOG_ERROR_HINTS)
        and not _should_ignore_log_line(line)
    ]


def analyze_log_text(
    text: str,
    top_k: int = 12,
    min_count: int = 2,
    max_updates: int = 12,
) -> dict[str, Any]:
    lines = _extract_error_lines_from_text(text)
    result = analyze_error_messages(
        lines,
        top_k=top_k,
        min_count=min_count,
        max_updates=max_updates,
    )
    result["error_line_count"] = len(lines)
    result["signature_patch_template"] = render_signature_patch_template(result["signature_update_plan"])
    return result


def analyze_log_file(
    log_path: str | Path,
    top_k: int = 12,
    min_count: int = 2,
    max_updates: int = 12,
    encoding: str = "utf-8",
) -> dict[str, Any]:
    path = Path(log_path)
    text = path.read_text(encoding=encoding, errors="ignore")
    result = analyze_log_text(text, top_k=top_k, min_count=min_count, max_updates=max_updates)
    result["log_path"] = str(path)
    return result


def get_compat_policy() -> dict[str, str]:
    return {"mode": _COMPAT_POLICY_MODE}


def set_compat_policy(mode: str | None) -> str:
    global _COMPAT_POLICY_MODE
    _COMPAT_POLICY_MODE = _normalize_policy_mode(mode)
    return _COMPAT_POLICY_MODE


def _ensure_op_stats(op_name: str) -> dict[str, Any]:
    return _OP_RUNTIME_STATS.setdefault(
        op_name,
        {
            "attempts": 0,
            "primary_success": 0,
            "primary_failures": 0,
            "fallback_count": 0,
            "primary_time_ms": 0.0,
            "fallback_time_ms": 0.0,
            "last_error": None,
            "last_error_class": None,
        },
    )


def reset_fallback_stats() -> None:
    _OP_RUNTIME_STATS.clear()
    _WARNED_FALLBACKS.clear()


def get_fallback_stats() -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for op_name, stats in _OP_RUNTIME_STATS.items():
        attempts = int(stats.get("attempts", 0))
        fallback_count = int(stats.get("fallback_count", 0))
        output[op_name] = {
            "count": fallback_count,
            "attempts": attempts,
            "fallback_ratio": (float(fallback_count) / float(attempts)) if attempts > 0 else 0.0,
            "primary_success": int(stats.get("primary_success", 0)),
            "primary_failures": int(stats.get("primary_failures", 0)),
            "primary_time_ms": round(float(stats.get("primary_time_ms", 0.0)), 3),
            "fallback_time_ms": round(float(stats.get("fallback_time_ms", 0.0)), 3),
            "last_error": stats.get("last_error"),
            "last_error_class": stats.get("last_error_class"),
            "recommended_action": recommended_action(str(stats.get("last_error_class", "unknown"))),
        }
    return output


def get_perf_counters() -> dict[str, Any]:
    ops = get_fallback_stats()
    total_attempts = sum(int(stats["attempts"]) for stats in ops.values())
    total_fallback = sum(int(stats["count"]) for stats in ops.values())
    total_primary_time_ms = sum(float(stats["primary_time_ms"]) for stats in ops.values())
    total_fallback_time_ms = sum(float(stats["fallback_time_ms"]) for stats in ops.values())
    return {
        "total_attempts": total_attempts,
        "total_fallback_count": total_fallback,
        "overall_fallback_ratio": (float(total_fallback) / float(total_attempts)) if total_attempts > 0 else 0.0,
        "total_primary_time_ms": round(total_primary_time_ms, 3),
        "total_fallback_time_ms": round(total_fallback_time_ms, 3),
    }


def _record_primary_success(op_name: str, elapsed_s: float) -> None:
    stats = _ensure_op_stats(op_name)
    stats["attempts"] = int(stats["attempts"]) + 1
    stats["primary_success"] = int(stats["primary_success"]) + 1
    stats["primary_time_ms"] = float(stats["primary_time_ms"]) + elapsed_s * 1000.0


def _record_primary_failure(op_name: str, exc: Exception, elapsed_s: float) -> None:
    stats = _ensure_op_stats(op_name)
    error_class = classify_runtime_error(exc)
    stats["attempts"] = int(stats["attempts"]) + 1
    stats["primary_failures"] = int(stats["primary_failures"]) + 1
    stats["primary_time_ms"] = float(stats["primary_time_ms"]) + elapsed_s * 1000.0
    stats["last_error"] = str(exc)
    stats["last_error_class"] = error_class


def _record_fallback_execution(op_name: str, elapsed_s: float) -> None:
    stats = _ensure_op_stats(op_name)
    stats["fallback_count"] = int(stats["fallback_count"]) + 1
    stats["fallback_time_ms"] = float(stats["fallback_time_ms"]) + elapsed_s * 1000.0


def _handle_runtime_failure(op_name: str, exc: Exception) -> None:
    mode = _COMPAT_POLICY_MODE
    if mode == "strict":
        raise exc
    if mode == "warn":
        warned_key = (op_name, classify_runtime_error(exc), str(exc))
        if warned_key not in _WARNED_FALLBACKS:
            warnings.warn(
                f"npu_compat fallback for {op_name}: {exc}",
                category=RuntimeWarning,
                stacklevel=3,
            )
            _WARNED_FALLBACKS.add(warned_key)


def _run_with_policy(
    op_name: str,
    primary_fn: Callable[[], Any],
    fallback_fn: Callable[[], Any],
) -> Any:
    primary_start = time.perf_counter()
    try:
        result = primary_fn()
    except Exception as exc:
        primary_elapsed = time.perf_counter() - primary_start
        _record_primary_failure(op_name, exc, primary_elapsed)
        _handle_runtime_failure(op_name, exc)
        fallback_start = time.perf_counter()
        fallback_result = fallback_fn()
        fallback_elapsed = time.perf_counter() - fallback_start
        _record_fallback_execution(op_name, fallback_elapsed)
        return fallback_result
    primary_elapsed = time.perf_counter() - primary_start
    _record_primary_success(op_name, primary_elapsed)
    return result


@lru_cache(maxsize=1)
def runtime_info() -> dict[str, Any]:
    info = {
        "torch_version": torch.__version__,
        "torch_npu_version": None,
        "npu_available": False,
        "device_count": 0,
    }
    if hasattr(torch, "npu"):
        try:
            info["npu_available"] = bool(torch.npu.is_available())
            if info["npu_available"]:
                info["device_count"] = int(torch.npu.device_count())
        except Exception:
            info["npu_available"] = False
    try:
        import torch_npu  # type: ignore

        info["torch_npu_version"] = getattr(torch_npu, "__version__", None)
    except Exception:
        info["torch_npu_version"] = None
    return info


def _probe_nonzero(device: torch.device) -> None:
    tensor = torch.tensor([0, 1, 0, 1], device=device)
    torch.nonzero(tensor)


def _probe_any(device: torch.device) -> None:
    tensor = torch.tensor([0, 0, 1], device=device)
    _ = tensor.bool().any()


def _probe_topk(device: torch.device) -> None:
    tensor = torch.randn(4, 8, device=device)
    torch.topk(tensor, k=2, dim=-1)


def _probe_softmax(device: torch.device) -> None:
    tensor = torch.randn(4, 8, device=device)
    torch.softmax(tensor, dim=-1)


_PROBE_TABLE: dict[str, ProbeFn] = {
    "nonzero": _probe_nonzero,
    "any": _probe_any,
    "topk": _probe_topk,
    "softmax": _probe_softmax,
}


@lru_cache(maxsize=64)
def supports_op(op_name: str, device_type: str = "auto") -> bool:
    probe = _PROBE_TABLE.get(op_name)
    if probe is None:
        return False

    if device_type == "auto":
        device = _resolve_device()
    else:
        device = _resolve_device(torch.device(device_type))

    try:
        probe(device)
        return True
    except Exception:
        return False


def compatibility_report(device_type: str = "auto") -> dict[str, Any]:
    info = runtime_info()
    if device_type == "auto":
        resolved_device_type = _resolve_device().type
    elif device_type in {"cpu", "npu"}:
        resolved_device_type = "npu" if device_type == "npu" and info["npu_available"] else "cpu"
    else:
        resolved_device_type = "cpu"
    report = {
        "runtime": info,
        "device_type": resolved_device_type,
        "policy": get_compat_policy(),
        "error_signatures": {
            "patterns": known_error_signatures(),
            "recommendations": dict(_ERROR_RECOMMENDATIONS),
        },
        "ops": {},
        "fallback_stats": get_fallback_stats(),
        "perf_counters": get_perf_counters(),
    }
    probe_device_type = report["device_type"]
    for op_name in _PROBE_TABLE:
        report["ops"][op_name] = supports_op(op_name, device_type=probe_device_type)
    report["fallback_stats"] = get_fallback_stats()
    report["perf_counters"] = get_perf_counters()
    return report


def safe_topk(values: torch.Tensor, k: int, dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    def _primary() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.topk(values, k=k, dim=dim)

    def _fallback() -> tuple[torch.Tensor, torch.Tensor]:
        sorted_values, sorted_indices = torch.sort(values, dim=dim, descending=True)
        return sorted_values.narrow(dim, 0, k), sorted_indices.narrow(dim, 0, k)

    return _run_with_policy("topk", _primary, _fallback)


def safe_softmax(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    def _primary() -> torch.Tensor:
        return torch.softmax(values, dim=dim)

    def _fallback() -> torch.Tensor:
        cpu_values = values.detach().cpu()
        cpu_result = F.softmax(cpu_values, dim=dim)
        return cpu_result.to(values.device)

    return _run_with_policy("softmax", _primary, _fallback)


def safe_has_any_tokens(values: torch.Tensor) -> bool:
    def _primary() -> bool:
        return bool(values.bool().any().item())

    def _fallback() -> bool:
        total = values.float().sum()
        return _to_python_scalar(total) > 0.0

    return bool(_run_with_policy("any", _primary, _fallback))


def safe_nonzero(mask: torch.Tensor) -> torch.Tensor:
    def _primary() -> torch.Tensor:
        return torch.nonzero(mask, as_tuple=False)

    def _fallback() -> torch.Tensor:
        flat = mask.reshape(-1).bool()
        idx = torch.arange(flat.numel(), device=mask.device)[flat]
        return idx.unsqueeze(-1)

    return _run_with_policy("nonzero", _primary, _fallback)
