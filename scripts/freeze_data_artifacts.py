"""Plan data artifact freeze metadata without materializing paper-scale splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.data.artifact_freeze import plan_data_artifact_freeze  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--output-dir", default="outputs/launch/paper_v1/protocol")
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()
    manifest = plan_data_artifact_freeze(args.config, args.output_dir, materialize=args.materialize)
    print(json.dumps({"datasets": len(manifest["datasets"]), "status": manifest["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
