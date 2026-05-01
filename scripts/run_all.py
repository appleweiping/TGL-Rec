"""Phase 1 run-all wrapper.

For Phase 1, run_all delegates to the experiment runner, which performs preprocessing,
skeleton prediction generation, and evaluation under one resolved config.
"""

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
    print(f"run_all completed: {result.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
