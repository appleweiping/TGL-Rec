"""Filter converted Amazon multidomain JSONL artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.data.amazon_filtering import filter_amazon_multidomain  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.materialize:
        parser.error("--dry-run and --materialize are mutually exclusive")
    report = filter_amazon_multidomain(args.config, dry_run=args.dry_run, materialize=args.materialize)
    print(
        json.dumps(
            {
                "materialized": report.get("materialized", False),
                "output_interactions": report.get("output_interactions"),
                "output_items": report.get("output_items"),
                "output_users": report.get("output_users"),
                "raw_files_unchanged": report.get("raw_files_unchanged"),
                "status": report.get("status"),
                "strategy": report.get("filtering_strategy"),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
