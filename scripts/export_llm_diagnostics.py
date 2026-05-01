"""Export Phase 3A LLM diagnostic summaries from an existing run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.diagnostics.llm_sequence_time_runner import export_llm_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = export_llm_diagnostics(args.run_dir)
    print(f"LLM diagnostic summary written: {args.run_dir / 'llm_diagnostic_summary.json'}")
    print(f"summary keys: {', '.join(sorted(summary))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

