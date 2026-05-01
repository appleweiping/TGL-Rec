"""Thin preprocessing wrapper for llm4rec configs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.data.preprocess import preprocess_from_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = preprocess_from_config(args.config)
    print(f"processed dataset: {result.output_dir}")
    print(
        " ".join(
            [
                f"users={result.metadata['user_count']}",
                f"items={result.metadata['item_count']}",
                f"interactions={result.metadata['interaction_count']}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
