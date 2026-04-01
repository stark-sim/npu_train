#!/usr/bin/env python3
"""Print NPU compatibility report as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from npu_parallel.npu_compat import compatibility_report, set_compat_policy


def main() -> int:
    parser = argparse.ArgumentParser(description="Print NPU compatibility report")
    parser.add_argument("--device-type", choices=["auto", "cpu", "npu"], default="auto")
    parser.add_argument("--policy", choices=["fallback", "warn", "strict"], default=None)
    args = parser.parse_args()
    if args.policy is not None:
        set_compat_policy(args.policy)
    print(json.dumps(compatibility_report(device_type=args.device_type), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
