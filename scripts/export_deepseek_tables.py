"""Export separate Phase 9D API LLM paper tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.experiments.deepseek_llm import export_deepseek_tables  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    result = export_deepseek_tables(args.run_dir)
    print(f"deepseek API LLM table written: {result['table_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
