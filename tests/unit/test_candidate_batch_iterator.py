from __future__ import annotations

from pathlib import Path

import pytest

from llm4rec.io.artifacts import sha256_file, write_json, write_jsonl
from llm4rec.scoring.candidate_batch import CandidateBatchIterator


def test_candidate_batch_iterator_resolves_shared_pool_and_refs(tmp_path: Path) -> None:
    pool_path = tmp_path / "candidate_pool.json"
    candidates_path = tmp_path / "candidates.jsonl"
    write_json(
        pool_path,
        {
            "candidate_items": ["i1", "i2", "i3"],
            "candidate_size": 3,
            "negative_pool_for_targets_outside_pool": ["i1", "i2"],
        },
    )
    write_jsonl(
        candidates_path,
        [
            {
                "candidate_size": 3,
                "candidate_storage": "shared_pool",
                "split": "test",
                "target_item": "i3",
                "user_id": "u1",
            },
            {
                "candidate_size": 3,
                "candidate_storage": "shared_pool",
                "domain": "books",
                "split": "test",
                "target_item": "i9",
                "user_id": "u2",
            },
        ],
    )

    batches = list(
        CandidateBatchIterator(
            candidate_artifact_path=candidates_path,
            candidate_artifact_sha256=sha256_file(candidates_path),
            candidate_pool_path=pool_path,
            candidate_pool_sha256=sha256_file(pool_path),
            history_by_user={"u1": ["i1", "i3"], "u2": ["i2"]},
            timestamp_by_user={"u1": 10.0},
            batch_size=2,
        )
    )

    assert len(batches) == 1
    batch = batches[0]
    assert batch.user_ids == ["u1", "u2"]
    assert batch.histories == [["i1"], ["i2"]]
    assert batch.candidate_item_ids[0] == ["i1", "i2", "i3"]
    assert batch.candidate_item_ids[1] == ["i1", "i2", "i9"]
    assert batch.candidate_refs[0]["candidate_row_id"] == "test|u1|i3"
    assert batch.candidate_refs[0]["candidate_pool_sha256"] == sha256_file(pool_path)
    assert batch.prediction_timestamps == [10.0, None]


def test_candidate_batch_iterator_verifies_checksum(tmp_path: Path) -> None:
    candidates_path = tmp_path / "candidates.jsonl"
    write_jsonl(
        candidates_path,
        [{"candidate_items": ["i1"], "split": "test", "target_item": "i1", "user_id": "u1"}],
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        CandidateBatchIterator(
            candidate_artifact_path=candidates_path,
            candidate_artifact_sha256="0" * 64,
            history_by_user={},
            batch_size=1,
        )
