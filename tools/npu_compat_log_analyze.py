#!/usr/bin/env python3
"""Analyze training logs and propose reviewable npu_compat signature updates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from npu_parallel.npu_compat import analyze_log_file, build_signature_update_plan, render_signature_patch_template

LOG_SUFFIXES = {".log", ".out", ".txt", ".err", ".stderr", ".stdout"}
LOG_TEXT_HINTS = ("log", "trace", "error", "stderr", "stdout", "output", "train", "benchmark", "debug")


def looks_like_log_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in {".log", ".out", ".err", ".stderr", ".stdout"}:
        return True
    if suffix == ".txt" and any(hint in name for hint in LOG_TEXT_HINTS):
        return True
    return False


def expand_log_inputs(items: list[str], recursive: bool) -> list[Path]:
    collected: list[Path] = []
    for item in items:
        path = Path(item)
        if path.is_file():
            collected.append(path)
            continue
        if not path.is_dir():
            continue
        walker = path.rglob("*") if recursive else path.glob("*")
        for candidate in walker:
            if candidate.is_file() and candidate.suffix.lower() in LOG_SUFFIXES and looks_like_log_file(candidate):
                collected.append(candidate)
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in collected:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)
    return unique_paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze one or more logs for NPU/CANN/HCCL compatibility signatures"
    )
    parser.add_argument("log_files", nargs="+", help="Log files or directories")
    parser.add_argument("--top-k", type=int, default=12, help="Top candidate tokens for unknown signatures")
    parser.add_argument("--min-count", type=int, default=2, help="Minimum count to propose a signature update")
    parser.add_argument("--max-updates", type=int, default=12, help="Maximum number of proposed updates")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan directories for log-like files")
    args = parser.parse_args()

    input_paths = expand_log_inputs(args.log_files, recursive=args.recursive)

    reports = []
    total_class_counts: Counter[str] = Counter()
    unknown_token_counts: Counter[str] = Counter()

    for item in input_paths:
        report = analyze_log_file(
            item,
            top_k=args.top_k,
            min_count=args.min_count,
            max_updates=args.max_updates,
        )
        reports.append(report)
        total_class_counts.update(report.get("class_counts", {}))
        for candidate in report.get("unknown_signature_suggestions", []):
            token = str(candidate.get("candidate_pattern", ""))
            count = int(candidate.get("count", 0))
            if token:
                unknown_token_counts[token] += count

    aggregate_suggestions = [
        {"candidate_pattern": token, "count": int(count)}
        for token, count in unknown_token_counts.most_common(max(args.top_k, 0))
    ]
    aggregate_plan = build_signature_update_plan(
        total_class_counts,
        aggregate_suggestions,
        min_count=args.min_count,
        max_updates=args.max_updates,
    )
    aggregate_patch_template = render_signature_patch_template(aggregate_plan)
    output = {
        "inputs": [str(path) for path in input_paths],
        "files": reports,
        "aggregate": {
            "total_logs": len(reports),
            "class_counts": dict(total_class_counts),
            "unknown_signature_suggestions": aggregate_suggestions,
            "outcome_objective": aggregate_plan.get("objective"),
            "signature_update_plan": aggregate_plan,
            "signature_patch_template": aggregate_patch_template,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
