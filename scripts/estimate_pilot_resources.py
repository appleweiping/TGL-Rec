"""Estimate Phase 7 pilot resources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.experiments.resource_estimator import estimate_pilot_resources  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    estimate = estimate_pilot_resources(args.config)
    print(f"resource estimate written: {estimate['output_run_dir']}/resource_estimate.json")
    print(f"estimated_candidate_scores={estimate['estimated_candidate_scores']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
