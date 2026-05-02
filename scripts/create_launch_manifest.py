"""Create the Phase 8 paper launch manifest without executing jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.experiments.launch_manifest import create_launch_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/launch/paper_v1/launch_manifest.json")
    parser.add_argument("--protocol-version", default="protocol_v1")
    args = parser.parse_args()
    manifest = create_launch_manifest(args.output, protocol_version=args.protocol_version)
    print(json.dumps({"planned_runs": manifest["total_planned_runs"], "protocol_version": manifest["protocol_version"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
