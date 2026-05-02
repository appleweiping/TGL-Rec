from __future__ import annotations

import json
from pathlib import Path

from llm4rec.scoring.candidate_batch import CandidateBatch
from llm4rec.scoring.prediction_writer import CompactTopKPredictionWriter


def test_compact_topk_prediction_writer_omits_expanded_candidates(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    batch = CandidateBatch(
        user_ids=["u1"],
        histories=[["i1"]],
        target_items=["i3"],
        candidate_item_ids=[["i1", "i2", "i3"]],
        domains=["tiny"],
        candidate_refs=[
            {
                "artifact_id": "tiny_candidates_protocol_v1",
                "artifact_path": str(tmp_path / "candidates.jsonl"),
                "artifact_sha256": "abc",
                "candidate_row_id": "test|u1|i3",
                "candidate_size": 3,
            }
        ],
        candidate_rows=[{}],
        prediction_timestamps=[None],
    )

    with CompactTopKPredictionWriter(
        predictions_path,
        method="bm25",
        top_n_to_save=2,
        base_metadata={"seed": 0},
    ) as writer:
        rows = writer.write_batch(
            batch=batch,
            score_matrix=[[0.2, 0.9, 0.9]],
            scorer_name="bm25_shared_pool",
        )

    row = json.loads(predictions_path.read_text(encoding="utf-8").strip())
    assert rows[0]["predicted_items"] == ["i2", "i3"]
    assert "candidate_items" not in row
    assert row["candidate_ref"]["candidate_row_id"] == "test|u1|i3"
    assert row["metadata"]["candidate_schema"] == "compact_ref_v1"
    assert row["metadata"]["scorer"] == "bm25_shared_pool"
    assert len(row["predicted_items"]) == 2
    assert len(row["scores"]) == 2
