"""Run local 8B LoRA adapter reranking evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.evaluation.lora_rerank import run_lora_rerank_eval  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--base-model-path", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-m", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_lora_rerank_eval(
        args.config,
        base_model_path=args.base_model_path,
        limit=args.limit,
        split=args.split,
        top_m=args.top_m,
        dry_run=args.dry_run,
    )
    print(
        "lora rerank eval completed: "
        f"predictions={result['manifest']['num_predictions']} "
        f"status={result['manifest']['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
