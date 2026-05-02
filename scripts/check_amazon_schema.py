"""Inspect Amazon Reviews 2023 raw domain schemas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.data.amazon_reviews_2023 import inspect_amazon_reviews_2023  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_amazon_reviews_2023(args.config, args.output)
    summary = {
        "domains": {
            domain: {
                "can_convert": row["can_convert"],
                "items": row["metadata_file_candidate"],
                "reviews": row["review_file_candidate"],
                "status": row["status"],
            }
            for domain, row in report["domains"].items()
        },
        "overall_status": report["overall_status"],
    }
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
