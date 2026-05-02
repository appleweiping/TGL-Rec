"""Run a protocol-v1 paper matrix from frozen artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.experiments.paper_matrix import PaperMatrixRequest, normalize_method, run_paper_matrix  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--candidate-output-mode", choices=["expanded", "compact_ref"], default="expanded")
    parser.add_argument("--max-expanded-candidate-items", type=int, default=200)
    parser.add_argument("--rerun-failed-only", action="store_true")
    args = parser.parse_args()
    run_dir = run_paper_matrix(
        PaperMatrixRequest(
            manifest_path=args.manifest,
            matrix=str(args.matrix),
            seed=int(args.seed),
            datasets=tuple(str(dataset) for dataset in args.datasets),
            methods=tuple(normalize_method(str(method)) for method in args.methods),
            output_dir=args.output_dir,
            continue_on_failure=bool(args.continue_on_failure),
            candidate_output_mode="compact_ref_v1" if args.candidate_output_mode == "compact_ref" else "expanded",
            max_expanded_candidate_items=int(args.max_expanded_candidate_items),
            rerun_failed_only=bool(args.rerun_failed_only),
        )
    )
    print(f"paper matrix completed: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
