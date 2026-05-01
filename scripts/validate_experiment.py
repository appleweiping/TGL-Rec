"""Validate one experiment config before real runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.experiments.validate import validate_experiment_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = validate_experiment_config(args.config)
    print(f"experiment validation: {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
