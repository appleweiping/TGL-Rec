"""Create Phase 8 planned paper job queue without running jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.experiments.job_queue import create_job_queue, load_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", default="outputs/launch/paper_v1")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    jobs = create_job_queue(manifest, args.output_dir)
    print(json.dumps({"jobs": len(jobs), "status": "planned"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
