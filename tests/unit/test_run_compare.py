from pathlib import Path

import pytest

from llm4rec.evaluation.run_compare import PredictionRunSpec, compare_prediction_runs
from llm4rec.io.artifacts import write_jsonl


def _prediction(
    *,
    event_id: str,
    method: str,
    predicted_items: list[str],
) -> dict[str, object]:
    return {
        "candidate_items": ["i1", "i2", "i3"],
        "dataset": "books",
        "event_id": event_id,
        "method": method,
        "predicted_items": predicted_items,
        "source_event_id": event_id,
        "split": "test",
        "target_item": "i2",
        "user_id": f"u{event_id}",
    }


def test_compare_prediction_runs_exports_aligned_deltas(tmp_path: Path):
    base = tmp_path / "base.jsonl"
    ours = tmp_path / "ours.jsonl"
    write_jsonl(base, [_prediction(event_id="e1", method="base", predicted_items=["i1", "i2", "i3"])])
    write_jsonl(ours, [_prediction(event_id="e1", method="ours", predicted_items=["i2", "i1", "i3"])])

    result = compare_prediction_runs(
        runs=[
            PredictionRunSpec("base", base),
            PredictionRunSpec("ours", ours),
        ],
        output_dir=tmp_path / "compare",
        baseline="base",
        ks=(1, 2),
    )

    assert result["manifest"]["num_aligned_events"] == 1
    event_rows = (tmp_path / "compare" / "aligned_event_comparison.csv").read_text(encoding="utf-8")
    assert "delta_hit@1_vs_base" in event_rows
    assert (tmp_path / "compare" / "method_summary.csv").is_file()


def test_compare_prediction_runs_strict_rejects_missing_events(tmp_path: Path):
    base = tmp_path / "base.jsonl"
    ours = tmp_path / "ours.jsonl"
    write_jsonl(base, [_prediction(event_id="e1", method="base", predicted_items=["i1", "i2"])])
    write_jsonl(ours, [_prediction(event_id="e2", method="ours", predicted_items=["i2", "i1"])])

    with pytest.raises(ValueError, match="not aligned"):
        compare_prediction_runs(
            runs=[
                PredictionRunSpec("base", base),
                PredictionRunSpec("ours", ours),
            ],
            output_dir=tmp_path / "compare",
        )
