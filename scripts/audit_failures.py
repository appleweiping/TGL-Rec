"""Audit pilot method failures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.evaluation.failure_audit import audit_failures  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    report = audit_failures(args.run_dir)
    print(f"failure_report={args.run_dir / 'failure_report.json'}")
    print(f"failure_count={report['failure_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
