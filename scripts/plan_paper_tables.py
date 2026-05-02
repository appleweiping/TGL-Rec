"""Plan paper table shells without metric values."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.evaluation.table_plan import plan_paper_tables  # noqa: E402
from llm4rec.experiments.job_queue import load_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", default="outputs/launch/paper_v1/table_plan.json")
    args = parser.parse_args()
    plan = plan_paper_tables(load_manifest(args.manifest), args.output)
    print(json.dumps({"numeric_values_present": plan["numeric_values_present"], "tables": len(plan["tables"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
