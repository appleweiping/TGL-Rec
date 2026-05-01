"""Thin wrapper for Phase 5 method smoke runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.methods.time_graph_evidence import run_time_graph_evidence_smoke  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = run_time_graph_evidence_smoke(args.config)
    print(f"method smoke run: {result.run_dir}")
    print(f"num_predictions={result.metrics.get('num_predictions', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
