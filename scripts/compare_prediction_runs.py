"""Compare aligned prediction JSONL runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.evaluation.run_compare import PredictionRunSpec, compare_prediction_runs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in name=/path/to/predictions.jsonl form. Provide at least two.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--k", action="append", type=int, dest="ks")
    parser.add_argument("--allow-missing-events", action="store_true")
    args = parser.parse_args()

    specs = [_parse_run_spec(value) for value in args.run]
    result = compare_prediction_runs(
        runs=specs,
        output_dir=args.output_dir,
        baseline=args.baseline,
        ks=tuple(args.ks or [1, 5, 10]),
        strict=not args.allow_missing_events,
    )
    print(
        "compared prediction runs: "
        f"runs={len(specs)} aligned_events={result['manifest']['num_aligned_events']} "
        f"output={args.output_dir}"
    )
    return 0


def _parse_run_spec(value: str) -> PredictionRunSpec:
    if "=" not in value:
        raise ValueError("--run must be in name=/path/to/predictions.jsonl form")
    name, path = value.split("=", 1)
    if not name:
        raise ValueError("--run name cannot be empty")
    return PredictionRunSpec(name=name, path=Path(path))


if __name__ == "__main__":
    raise SystemExit(main())
