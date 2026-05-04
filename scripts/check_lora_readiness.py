"""Check local 8B LoRA/QLoRA readiness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.trainers.gpu_guard import check_lora_readiness  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    report = check_lora_readiness(args.config)
    print(
        "lora readiness: "
        f"feasible={report.get('feasible')} "
        f"cuda={report.get('cuda_available')} "
        f"model_path_exists={report.get('base_model_path_exists')} "
        f"blockers={report.get('blockers', [])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
