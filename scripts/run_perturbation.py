"""Run Phase 2B sequence/time perturbation diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.diagnostics.perturbation_runner import run_perturbation_experiment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_dir = run_perturbation_experiment(args.config)
    print(f"perturbation diagnostics written: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
