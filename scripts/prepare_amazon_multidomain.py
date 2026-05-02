"""Prepare Amazon Reviews 2023 multidomain JSONL artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.data.amazon_converter import prepare_amazon_multidomain  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if sum(bool(value) for value in [args.dry_run, args.materialize, args.preflight]) > 1:
        parser.error("--preflight, --dry-run, and --materialize are mutually exclusive")
    report = prepare_amazon_multidomain(args.config, dry_run=args.dry_run, materialize=args.materialize, preflight=args.preflight)
    print(
        json.dumps(
            {
                "materialized": report.get("materialized", False),
                "mode": report.get("mode", report.get("conversion_mode")),
                "status": report.get("status", "PREFLIGHT"),
                "summary": report.get("summary", {}),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
