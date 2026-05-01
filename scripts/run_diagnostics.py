"""Run Phase 2A sequence/time diagnostic artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.diagnostics.runner import run_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_dir = run_diagnostics(args.config)
    print(f"diagnostics written: {run_dir / 'diagnostics'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
