"""Build train-only SFT data for local LoRA experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.trainers.sft_dataset import build_lora_sft_data  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()
    results = build_lora_sft_data(args.config, dry_run=args.dry_run, materialize=args.materialize)
    for result in results:
        print(
            "sft data: "
            f"dataset={result.manifest['dataset']} "
            f"variant={result.manifest['variant']} "
            f"train={result.manifest['num_train_rows']} "
            f"valid={result.manifest['num_valid_rows']} "
            f"leakage_free={result.leakage_audit['leakage_free']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
