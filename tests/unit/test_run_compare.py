from pathlib import Path

import pytest

from llm4rec.evaluation.run_compare import PredictionRunSpec, compare_prediction_runs
from llm4rec.io.artifacts import write_jsonl


def _prediction(
    *,
    event_id: str,
    method: str,
    predicted_items: list[str],
    candidate_items: list[str] | None = None,
    metadata: dict[str, object] | None = None,
    protocol_version: str = "protocol_week8_large10000_same_candidate",
) -> dict[str, object]:
    return {
        "candidate_items": candidate_items or ["i1", "i2", "i3"],
        "dataset": "books",
        "event_id": event_id,
        "metadata": metadata or {},
        "method": method,
        "predicted_items": predicted_items,
        "protocol_version": protocol_version,
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
        allow_non_reportable=False,
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


def test_compare_prediction_runs_rejects_candidate_mismatch(tmp_path: Path):
    base = tmp_path / "base.jsonl"
    ours = tmp_path / "ours.jsonl"
    write_jsonl(base, [_prediction(event_id="e1", method="base", predicted_items=["i1", "i2"])])
    write_jsonl(
        ours,
        [
            _prediction(
                event_id="e1",
                method="ours",
                predicted_items=["i2", "i1"],
                candidate_items=["i2", "i1", "i3"],
            )
        ],
    )

    with pytest.raises(ValueError, match="Candidate set mismatch"):
        compare_prediction_runs(
            runs=[
                PredictionRunSpec("base", base),
                PredictionRunSpec("ours", ours),
            ],
            output_dir=tmp_path / "compare",
        )


def test_compare_prediction_runs_rejects_non_reportable_without_override(tmp_path: Path):
    base = tmp_path / "base.jsonl"
    ours = tmp_path / "ours.jsonl"
    write_jsonl(base, [_prediction(event_id="e1", method="base", predicted_items=["i1", "i2"])])
    write_jsonl(
        ours,
        [
            _prediction(
                event_id="e1",
                method="ours",
                predicted_items=["i2", "i1"],
                metadata={"do_not_merge_into_main_accuracy_table": True},
            )
        ],
    )

    with pytest.raises(ValueError, match="non-reportable"):
        compare_prediction_runs(
            runs=[
                PredictionRunSpec("base", base),
                PredictionRunSpec("ours", ours),
            ],
            output_dir=tmp_path / "compare",
        )

    compare_prediction_runs(
        runs=[
            PredictionRunSpec("base", base),
            PredictionRunSpec("ours", ours),
        ],
        output_dir=tmp_path / "compare_allowed",
        allow_non_reportable=True,
    )
