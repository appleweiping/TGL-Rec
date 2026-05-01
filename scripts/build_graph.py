"""Build Phase 2A graph diagnostic artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.diagnostics.runner import build_graph_artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_dir = build_graph_artifacts(args.config)
    print(f"graph diagnostics written: {run_dir / 'diagnostics'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
