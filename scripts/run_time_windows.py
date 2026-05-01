"""Run Phase 2B time-window graph diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.diagnostics.time_window_runner import run_time_window_experiment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_dir = run_time_window_experiment(args.config)
    print(f"time-window diagnostics written: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
