"""Estimate Phase 3B API micro diagnostic cost before any API call."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.diagnostics.api_micro_runner import estimate_api_micro_cost  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_dir, _preflight = estimate_api_micro_cost(args.config)
    print(f"Cost preflight written: {run_dir / 'cost_preflight.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
