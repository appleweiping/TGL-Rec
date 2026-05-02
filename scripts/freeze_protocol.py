"""Freeze protocol metadata for Phase 8 launch preparation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.experiments.protocol_version import freeze_protocol  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--output-dir", default="outputs/launch/paper_v1/protocol")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--force-new-version", action="store_true")
    args = parser.parse_args()
    manifest = freeze_protocol(
        args.version,
        args.output_dir,
        dry_run=bool(args.dry_run or not args.materialize),
        materialize=args.materialize,
        force_new_version=args.force_new_version,
    )
    print(json.dumps({"protocol_version": manifest["protocol_version"], "status": manifest["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
