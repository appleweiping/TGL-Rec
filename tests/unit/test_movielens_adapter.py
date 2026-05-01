from __future__ import annotations

from pathlib import Path

import pytest

from llm4rec.data.movielens_adapter import (
    MISSING_MOVIELENS_MESSAGE,
    load_movielens_style,
    preprocess_movielens_from_config,
)


def test_movielens_adapter_reads_raw_fixture_and_writes_artifacts() -> None:
    root = Path(__file__).resolve().parents[2]
    output_dir = root / "outputs" / "test_runs" / "unit_movielens_adapter" / "processed"
    result = preprocess_movielens_from_config(
        {
            "dataset": {
                "adapter": "movielens_style",
                "candidate_protocol": "full_catalog",
                "min_user_interactions": 3,
                "name": "fixture_ml",
                "output_dir": str(output_dir),
                "paths": {"raw_dir": str(root / "data" / "fixtures" / "movielens_style")},
                "seed": 2026,
                "split_strategy": "leave_one_out",
            }
        }
    )
    assert result.metadata["user_count"] == 4
    assert result.metadata["item_count"] == 9
    assert (result.output_dir / "train.jsonl").is_file()
    assert (result.output_dir / "candidates.jsonl").is_file()


def test_movielens_adapter_missing_data_error_is_actionable() -> None:
    root = Path(__file__).resolve().parents[2]
    with pytest.raises(FileNotFoundError, match="MovieLens-style data is missing"):
        load_movielens_style({"raw_dir": str(root / "outputs" / "missing_movielens")})
    assert "ratings.dat" in MISSING_MOVIELENS_MESSAGE
