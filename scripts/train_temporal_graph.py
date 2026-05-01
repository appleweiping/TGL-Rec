"""Train/evaluate TemporalGraphEncoder smoke config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.trainers.temporal_graph import run_temporal_graph_smoke  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = run_temporal_graph_smoke(args.config)
    print(f"TemporalGraphEncoder smoke status: {result.status}")
    print(f"run_dir={result.run_dir}")
    print(f"checkpoint={result.checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
