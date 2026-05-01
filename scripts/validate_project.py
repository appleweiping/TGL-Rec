"""CI-style pre-experiment project validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.experiments.validate import validate_project  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=ROOT, type=Path)
    args = parser.parse_args()
    result = validate_project(args.root)
    print(f"project validation: {result['status']}")
    print(f"checked_files={len(result['checked_files'])}")
    optional = result.get("optional_dependencies", {})
    if optional.get("torch") is False:
        print("optional dependency torch: unavailable (SASRec/TemporalGraphEncoder smoke training will be skipped)")
    elif optional.get("torch") is True:
        print("optional dependency torch: available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
