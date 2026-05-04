"""Train local 8B LoRA/QLoRA adapters when readiness passes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.trainers.lora_sft import train_lora_8b  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = train_lora_8b(args.config, dry_run=args.dry_run)
    print(f"lora training status: {result.get('status')} variant={result.get('variant')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
