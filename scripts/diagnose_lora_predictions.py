"""Diagnose local LoRA reranking prediction artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.evaluation.lora_diagnostics import diagnose_lora_predictions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = diagnose_lora_predictions(args.predictions, args.output_dir)
    print(
        "lora diagnostics completed: "
        f"rows={result['overall']['num_rows']} "
        f"prompt_continuation_rate={result['overall']['prompt_continuation_rate']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
