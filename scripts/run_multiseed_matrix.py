"""Run Phase 9C nonzero paper matrix seeds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.experiments.multiseed_paper import run_multiseed_paper_matrix  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-output-mode", choices=["expanded", "compact_ref"], default="expanded")
    parser.add_argument("--shared-pool-scoring", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--max-expanded-candidate-items", type=int, default=200)
    parser.add_argument("--scoring-config", type=Path, default=Path("configs/scoring/shared_pool.yaml"))
    args = parser.parse_args()
    run_dir = run_multiseed_paper_matrix(
        manifest_path=args.manifest,
        matrix=str(args.matrix),
        seeds=[int(seed) for seed in args.seeds],
        datasets=[str(dataset) for dataset in args.datasets],
        methods=[str(method) for method in args.methods],
        output_dir=args.output_dir,
        candidate_output_mode=str(args.candidate_output_mode),
        shared_pool_scoring=bool(args.shared_pool_scoring),
        continue_on_failure=bool(args.continue_on_failure),
        max_expanded_candidate_items=int(args.max_expanded_candidate_items),
        scoring_config_path=args.scoring_config,
    )
    print(f"multiseed matrix completed: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
