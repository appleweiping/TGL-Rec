"""Lock a completed run after verifying required result artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.evaluation.result_lock import lock_results  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = lock_results(args.run_dir, args.output)
    print(json.dumps({"locked": manifest["locked"], "run_dir": manifest["run_dir"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
