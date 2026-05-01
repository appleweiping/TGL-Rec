"""Train/evaluate SASRec smoke config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.trainers.sasrec import run_sasrec_smoke  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = run_sasrec_smoke(args.config)
    print(f"SASRec smoke status: {result.status}")
    print(f"run_dir={result.run_dir}")
    print(f"checkpoint={result.checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
