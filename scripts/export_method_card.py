"""Export the Phase 5 method card from config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.methods.time_graph_evidence import export_method_card  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", default=None, type=Path)
    args = parser.parse_args()
    path = export_method_card(args.config, args.output)
    print(f"method card written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
