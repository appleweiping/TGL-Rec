"""Export Phase 9B main-accuracy seed=0 tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.experiments.paper_matrix import export_main_accuracy_tables  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    result = export_main_accuracy_tables(args.run_dir)
    print(f"main accuracy table written: {result['table_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
