"""Run Phase 7 NON_REPORTABLE pilot matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.experiments.pilot_runner import run_pilot_matrix  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_dir = run_pilot_matrix(args.config)
    print(f"pilot matrix completed: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
