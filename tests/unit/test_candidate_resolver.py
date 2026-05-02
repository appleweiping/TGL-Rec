from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm4rec.data.candidates import candidate_row_id
from llm4rec.evaluation.candidate_resolver import CandidateResolutionError, CandidateResolver
from llm4rec.io.artifacts import sha256_file, write_json, write_jsonl


def test_candidate_resolver_resolves_expanded_row_and_writes_index(tmp_path: Path) -> None:
    candidates_path = tmp_path / "candidates.jsonl"
    row_id = candidate_row_id(user_id="u1", target_item="i3", split="test")
    write_jsonl(
        candidates_path,
        [
            {
                "candidate_row_id": row_id,
                "candidate_items": ["i1", "i3"],
                "split": "test",
                "target_item": "i3",
                "user_id": "u1",
            }
        ],
    )
    resolver = CandidateResolver(
        candidate_artifact_path=candidates_path,
        candidate_artifact_sha256=sha256_file(candidates_path),
    )

    assert resolver.get_candidates(candidate_row_id_value=row_id, target_item="i3") == ["i1", "i3"]
    assert (tmp_path / "candidate_index.json").is_file()


def test_candidate_resolver_resolves_shared_pool_target_outside_pool(tmp_path: Path) -> None:
    candidates_path = tmp_path / "candidates.jsonl"
    pool_path = tmp_path / "candidate_pool.json"
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
                "candidate_pool_artifact": str(pool_path),
                "candidate_size": 3,
                "candidate_storage": "shared_pool",
                "split": "test",
                "target_item": "i9",
                "user_id": "u1",
            }
        ],
    )
    ref = {
        "artifact_id": "tiny",
        "artifact_path": str(candidates_path),
        "artifact_sha256": sha256_file(candidates_path),
        "candidate_pool_artifact": str(pool_path),
        "candidate_pool_sha256": sha256_file(pool_path),
        "candidate_row_id": "test|u1|i9",
        "candidate_size": 3,
        "candidate_storage": "shared_pool",
    }

    resolver = CandidateResolver.from_ref(ref)

    assert resolver.get_candidates(candidate_row_id_value="test|u1|i9", target_item="i9", candidate_ref=ref) == ["i1", "i2", "i9"]


def test_candidate_resolver_stops_on_checksum_mismatch(tmp_path: Path) -> None:
    candidates_path = tmp_path / "candidates.jsonl"
    candidates_path.write_text(json.dumps({"user_id": "u1"}) + "\n", encoding="utf-8")

    with pytest.raises(CandidateResolutionError):
        CandidateResolver(candidate_artifact_path=candidates_path, candidate_artifact_sha256="0" * 64)
