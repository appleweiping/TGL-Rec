"""Estimate paper-scale resources from a launch manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.experiments.job_queue import load_manifest  # noqa: E402
from llm4rec.experiments.resource_budget import estimate_paper_resources  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", default="outputs/launch/paper_v1/resource_budget.json")
    args = parser.parse_args()
    budget = estimate_paper_resources(load_manifest(args.manifest), args.output)
    print(json.dumps({"api_calls": budget["api_calls"], "jobs": budget["jobs"], "lora_training_jobs": budget["lora_training_jobs"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
