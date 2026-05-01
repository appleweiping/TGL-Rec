import json
from pathlib import Path

from llm4rec.diagnostics.api_case_sampler import sample_api_micro_cases
from llm4rec.io.artifacts import write_jsonl


def test_api_case_sampler_builds_deterministic_grouped_samples(tmp_path):
    source = _write_source_artifacts(tmp_path)
    config = {
        "experiment": {"seed": 7},
        "diagnostic": {
            "case_groups": ["semantic_and_transition", "high_time_window_strength"],
            "candidate_size": 3,
            "max_cases_per_group": 1,
            "max_total_cases": 2,
            "random_seed": 7,
            "require_target_in_candidates": True,
        },
    }
    first = sample_api_micro_cases(source_run_dir=source, config=config)
    second = sample_api_micro_cases(source_run_dir=source, config=config)
    assert first == second
    assert {row["case_group"] for row in first} == {
        "semantic_and_transition",
        "high_time_window_strength",
    }
    assert all(len(row["candidate_items"]) <= 3 for row in first)
    assert all(row["target_item"] in row["candidate_items"] for row in first)


def _write_source_artifacts(tmp_path: Path) -> Path:
    source = tmp_path / "phase2b"
    processed = source / "artifacts" / "processed_dataset"
    diagnostics = source / "diagnostics"
    processed.mkdir(parents=True)
    diagnostics.mkdir(parents=True)
    write_jsonl(
        processed / "items.jsonl",
        [
            {"item_id": "i1", "title": "A", "raw_text": "A", "domain": "tiny"},
            {"item_id": "i2", "title": "B", "raw_text": "B", "domain": "tiny"},
            {"item_id": "i3", "title": "C", "raw_text": "C", "domain": "tiny"},
        ],
    )
    write_jsonl(
        processed / "interactions.jsonl",
        [
            {"user_id": "u1", "item_id": "i1", "timestamp": 1, "split": "train", "domain": "tiny"},
            {"user_id": "u1", "item_id": "i3", "timestamp": 2, "split": "test", "domain": "tiny"},
        ],
    )
    write_jsonl(
        processed / "candidates.jsonl",
        [
            {
                "user_id": "u1",
                "target_item": "i3",
                "candidate_items": ["i1", "i2", "i3"],
                "split": "test",
                "domain": "tiny",
            }
        ],
    )
    case = {
        "source_item": "i1",
        "target_item": "i2",
        "source_title": "A",
        "target_title": "B",
        "transition_count": 2,
        "transition_score": 2.0,
        "text_similarity": 0.2,
        "same_genre_or_category": False,
        "dominant_gap_bucket": "same_session",
        "time_window_score_1d": 5.0,
        "time_window_score_7d": 6.0,
        "time_window_score_30d": 6.5,
    }
    (diagnostics / "grouped_cases.json").write_text(
        json.dumps({"semantic_and_transition": [case]}, sort_keys=True),
        encoding="utf-8",
    )
    return source
