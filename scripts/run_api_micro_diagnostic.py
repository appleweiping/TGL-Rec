"""Run Phase 3B API micro diagnostics with a required dry-run option."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.diagnostics.api_micro_runner import run_api_micro_diagnostic  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_dir = run_api_micro_diagnostic(args.config, dry_run=args.dry_run)
    print(f"API micro diagnostic written: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
