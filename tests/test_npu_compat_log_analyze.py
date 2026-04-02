#!/usr/bin/env python3
"""CPU-only smoke tests for npu_compat log-analysis helpers."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from npu_parallel.npu_compat import (
    analyze_error_messages,
    analyze_log_file,
    analyze_log_text,
    render_signature_patch_template,
)


def main() -> None:
    synthetic_log = """
    [INFO] training starts
    RuntimeError: ACL error code 500002 in topk kernel
    HCCL all_to_all timeout during dispatch
    Some strange crash path with xpu_backend_magic_foo
    Some strange crash path with xpu_backend_magic_foo
    Traceback (most recent call last):
    RuntimeError: operator not implemented
    """
    benign_warning_log = """
    [W401] Warning: The oprator of ne is executed, Currently High Accuracy but Low Performance OP with 64-bit has been used
    [W401] Warning: [Check][offset] Check input storage_offset[%ld] = 0 failed, result is untrustworthy64
    RuntimeError: ACL error code 500002 in topk kernel
    """

    text_result = analyze_log_text(synthetic_log, top_k=5, min_count=1, max_updates=8)
    assert text_result["error_line_count"] >= 5
    assert text_result["class_counts"]["acl_runtime"] >= 1
    assert text_result["class_counts"]["hccl_runtime"] >= 1
    assert text_result["class_counts"]["unsupported_op"] >= 1
    assert text_result["unknown_count"] >= 1
    assert isinstance(text_result["unknown_signature_suggestions"], list)
    assert "outcome_objective" in text_result
    assert "signature_update_plan" in text_result
    assert "signature_patch_template" in text_result
    assert text_result["signature_update_plan"]["policy"]["min_count_threshold"] == 1

    benign_result = analyze_log_text(benign_warning_log, top_k=4, min_count=1, max_updates=4)
    assert benign_result["error_line_count"] == 1
    assert benign_result["class_counts"]["acl_runtime"] == 1
    assert "shape_or_dtype" not in benign_result["class_counts"]

    message_result = analyze_error_messages(
        [
            "ACL runtime failure 500002",
            "OOM memory allocation failed",
            "xpu_backend_magic_foo crash happened",
            "xpu_backend_magic_foo crash happened",
        ],
        top_k=3,
        min_count=1,
        max_updates=5,
    )
    assert message_result["class_counts"]["acl_runtime"] == 1
    assert message_result["class_counts"]["memory"] == 1
    assert message_result["class_counts"]["unknown"] == 2
    assert "outcome_objective" in message_result
    assert "signature_update_plan" in message_result
    assert message_result["signature_update_plan"]["policy"]["max_updates"] == 5
    patch_template = render_signature_patch_template(message_result["signature_update_plan"])
    assert "*** Begin Patch" in patch_template or patch_template == ""

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        log_path = tmp_path / "train.log"
        nested_dir = tmp_path / "nested"
        nested_dir.mkdir()
        nested_log = nested_dir / "worker.stderr"
        ignored_txt = nested_dir / "merges.txt"
        log_path.write_text(synthetic_log, encoding="utf-8")
        nested_log.write_text("RuntimeError: HCCL timeout\nweird backend crash foo_magic\nweird backend crash foo_magic\n", encoding="utf-8")
        ignored_txt.write_text("_ error\n_ ERROR\n", encoding="utf-8")

        file_result = analyze_log_file(log_path, top_k=4, min_count=1, max_updates=6)
        assert file_result["log_path"].endswith("train.log")
        assert file_result["total_messages"] >= 5
        assert file_result["signature_update_plan"]["policy"]["max_updates"] == 6

        cli_result = subprocess.check_output(
            [
                sys.executable,
                str(ROOT / "tools" / "npu_compat_log_analyze.py"),
                str(tmp_path),
                "--recursive",
                "--top-k",
                "4",
                "--min-count",
                "1",
                "--max-updates",
                "6",
            ],
            text=True,
        )
        parsed = json.loads(cli_result)
        assert parsed["aggregate"]["total_logs"] == 2
        assert parsed["aggregate"]["class_counts"]["acl_runtime"] >= 1
        assert "outcome_objective" in parsed["aggregate"]
        assert "signature_update_plan" in parsed["aggregate"]
        assert "signature_patch_template" in parsed["aggregate"]
        assert len(parsed["inputs"]) == 2
        assert all(not item.endswith("merges.txt") for item in parsed["inputs"])

    print("NPU compatibility log-analysis smoke test: OK")


if __name__ == "__main__":
    main()
