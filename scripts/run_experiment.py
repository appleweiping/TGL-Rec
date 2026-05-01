"""Thin experiment runner wrapper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.experiments.runner import run_experiment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = run_experiment(args.config)
    print(f"run completed: {result.run_dir}")
    print(f"metrics: {result.run_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
