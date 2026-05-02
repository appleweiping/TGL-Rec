"""Check dataset readiness without downloading or running experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.data.readiness import check_dataset_readiness  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check_dataset_readiness(args.config, args.output)
    print(json.dumps({"dataset": report["dataset"], "status": report["status"], "blocker": report["blocker"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
