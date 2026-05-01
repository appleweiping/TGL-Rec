"""Export Phase 2B diagnostic summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.diagnostics.diagnostic_export import export_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--similarity-config",
        default=Path("configs/diagnostics/movielens_similarity_vs_transition.yaml"),
        type=Path,
    )
    args = parser.parse_args()
    summary = export_diagnostics(args.run_dir, similarity_config=args.similarity_config)
    print(f"diagnostic summary written: {args.run_dir / 'diagnostic_summary.json'}")
    print(f"summary keys: {', '.join(sorted(summary))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
