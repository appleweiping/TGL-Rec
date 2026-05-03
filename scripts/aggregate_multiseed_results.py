"""Aggregate Phase 9C multi-seed paper results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.experiments.multiseed_paper import aggregate_phase9c_outputs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed0-dir", required=True, type=Path)
    parser.add_argument("--multiseed-dir", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    args = parser.parse_args()
    result = aggregate_phase9c_outputs(
        seed0_dir=args.seed0_dir,
        multiseed_dir=args.multiseed_dir,
        seeds=[int(seed) for seed in args.seeds],
    )
    print(f"aggregate_metrics={result['aggregate_metrics']}")
    print(f"significance_tests={result['significance_tests']}")
    print(f"failure_count={result['failure_report']['failure_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
