from __future__ import annotations

from pathlib import Path

import pytest

from llm4rec.data.candidates import candidate_row_id
from llm4rec.evaluation.evaluator import evaluate_predictions
from llm4rec.io.artifacts import sha256_file, write_jsonl


def test_compact_and_expanded_predictions_have_identical_metrics(tmp_path: Path) -> None:
    items_path = tmp_path / "items.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    expanded_path = tmp_path / "expanded.jsonl"
    compact_path = tmp_path / "compact.jsonl"
    row_id = candidate_row_id(user_id="u1", target_item="i3", split="test")
    write_jsonl(
        items_path,
        [
            {"item_id": "i1", "title": "one", "description": None, "category": "a", "brand": None, "domain": "tiny", "raw_text": "one"},
            {"item_id": "i2", "title": "two", "description": None, "category": "a", "brand": None, "domain": "tiny", "raw_text": "two"},
            {"item_id": "i3", "title": "three", "description": None, "category": "b", "brand": None, "domain": "tiny", "raw_text": "three"},
        ],
    )
    write_jsonl(
        candidates_path,
        [
            {
                "candidate_row_id": row_id,
                "candidate_items": ["i1", "i2", "i3"],
                "split": "test",
                "target_item": "i3",
                "user_id": "u1",
            }
        ],
    )
    base = {
        "domain": "tiny",
        "method": "popularity",
        "predicted_items": ["i3", "i2"],
        "raw_output": None,
        "scores": [1.0, 0.2],
        "target_item": "i3",
        "user_id": "u1",
    }
    write_jsonl(expanded_path, [{**base, "candidate_items": ["i1", "i2", "i3"], "metadata": {}}])
    write_jsonl(
        compact_path,
        [
            {
                **base,
                "candidate_ref": {
                    "artifact_id": "tiny_candidates_protocol_v1",
                    "artifact_path": str(candidates_path),
                    "artifact_sha256": sha256_file(candidates_path),
                    "candidate_row_id": row_id,
                    "candidate_size": 3,
                },
                "metadata": {"candidate_schema": "compact_ref_v1"},
            }
        ],
    )

    expanded = evaluate_predictions(
        predictions_path=expanded_path,
        item_catalog_path=items_path,
        output_dir=tmp_path / "expanded_eval",
        ks=(1, 5, 10),
        candidate_protocol="fixed_sampled",
    )
    compact = evaluate_predictions(
        predictions_path=compact_path,
        item_catalog_path=items_path,
        output_dir=tmp_path / "compact_eval",
        ks=(1, 5, 10),
        candidate_protocol="fixed_sampled",
    )

    assert compact["candidate_schema"] == "compact_ref_v1"
    assert compact["overall"]["Recall@1"] == pytest.approx(expanded["overall"]["Recall@1"])
    assert compact["overall"]["validity_rate"] == pytest.approx(expanded["overall"]["validity_rate"])
