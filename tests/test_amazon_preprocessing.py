import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from tglrec.cli import main
from tglrec.data import schema
from tglrec.data.amazon import preprocess_amazon_reviews_2023
from tglrec.data.splits import assert_no_future_leakage
from tglrec.utils.io import read_json


def _write_jsonl_gz(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return path


def _write_synthetic_amazon(root: Path) -> tuple[Path, Path]:
    reviews = [
        {"user_id": "u2", "parent_asin": "B020", "asin": "B020A", "rating": 5, "timestamp": 100},
        {"user_id": "u2", "parent_asin": "B010", "asin": "B010A", "rating": 4, "timestamp": 110},
        {"user_id": "u2", "parent_asin": "B030", "asin": "B030A", "rating": 4, "timestamp": 120},
        {"user_id": "u1", "parent_asin": "B010", "asin": "B010A", "rating": 3, "timestamp": 130},
        {"user_id": "u1", "parent_asin": "B020", "asin": "B020A", "rating": 4, "timestamp": 140},
        {"user_id": "u1", "parent_asin": "B030", "asin": "B030A", "rating": 5, "timestamp": 150},
        {"user_id": "u3", "parent_asin": "B030", "asin": "B030A", "rating": 4, "timestamp": 160},
        {"user_id": "u3", "parent_asin": "B020", "asin": "B020A", "rating": 3, "timestamp": 170},
        {"user_id": "u3", "parent_asin": "B010", "asin": "B010A", "rating": 5, "timestamp": 180},
        {"user_id": "u4", "parent_asin": "B010", "asin": "B010A", "rating": 5, "timestamp": 190},
        {"user_id": "u4", "parent_asin": "B020", "asin": "B020A", "rating": 4, "timestamp": 200},
        {"user_id": "u4", "parent_asin": "B030", "asin": "B030A", "rating": 4, "timestamp": 210},
    ]
    metadata = [
        {
            "parent_asin": "B010",
            "title": "Cleanser",
            "main_category": "All Beauty",
            "categories": ["Beauty", "Skin Care"],
            "store": "Brand A",
            "description": ["daily cleanser"],
            "features": ["gentle"],
            "average_rating": 4.5,
            "rating_number": 10,
        },
        {
            "parent_asin": "B020",
            "title": "Toner",
            "main_category": "All Beauty",
            "categories": ["Beauty", "Skin Care"],
            "store": "Brand B",
            "description": ["hydrating toner"],
            "features": ["alcohol free"],
            "average_rating": 4.2,
            "rating_number": 8,
        },
        {
            "parent_asin": "B030",
            "title": "Moisturizer",
            "main_category": "All Beauty",
            "categories": ["Beauty", "Skin Care"],
            "store": "Brand C",
            "description": ["night cream"],
            "features": ["fragrance free"],
            "average_rating": 4.8,
            "rating_number": 12,
        },
    ]
    return (
        _write_jsonl_gz(root / "All_Beauty.jsonl.gz", reviews),
        _write_jsonl_gz(root / "meta_All_Beauty.jsonl.gz", metadata),
    )


def test_amazon_preprocessing_writes_temporal_splits(tmp_path: Path):
    reviews_path, metadata_path = _write_synthetic_amazon(tmp_path / "raw")
    output = tmp_path / "processed"

    result = preprocess_amazon_reviews_2023(
        reviews_path=reviews_path,
        metadata_path=metadata_path,
        category="all_beauty",
        output_dir=output,
        min_user_interactions=3,
        min_item_interactions=3,
        global_train_ratio=0.5,
        global_val_ratio=0.25,
        seed=2026,
    )

    interactions = pd.read_csv(output / "interactions.csv")
    assert result.num_interactions == 12
    assert result.metadata["dataset"] == "amazon_reviews_2023_all_beauty"
    assert (output / "config.yaml").exists()
    assert (output / "metadata.json").exists()
    assert (output / "checksums.json").exists()
    assert_no_future_leakage(interactions, schema.SPLIT_LOO)
    assert_no_future_leakage(interactions, schema.SPLIT_GLOBAL)
    metadata = read_json(output / "metadata.json")
    checksum_manifest = read_json(output / "checksums.json")
    assert metadata["same_user_timestamp_tie_stats"]["tied_extra_rows"] == 0
    assert metadata["same_user_timestamp_tie_stats_after_dedup"]["tied_extra_rows"] == 0
    assert metadata["processed_file_checksums"]["items.csv"]["sha256"]
    assert checksum_manifest["files"]["metadata.json"]["sha256"]

    per_user_counts = interactions.groupby(schema.USER_ID)[schema.SPLIT_LOO].value_counts()
    for user_id in interactions[schema.USER_ID].unique():
        assert per_user_counts[(user_id, "val")] == 1
        assert per_user_counts[(user_id, "test")] == 1

    items = pd.read_csv(output / "items.csv")
    assert set(items["title"]) == {"Cleanser", "Toner", "Moisturizer"}
    assert "[\"Beauty\", \"Skin Care\"]" in set(items["categories"])
    assert "average_rating" not in items.columns
    assert "rating_number" not in items.columns
    assert pd.read_csv(output / "temporal_leave_one_out" / "train.csv")[schema.TIMESTAMP].max() < 200


def test_amazon_cli_local_file_preprocessing(tmp_path: Path, capsys):
    reviews_path, metadata_path = _write_synthetic_amazon(tmp_path / "raw")
    output = tmp_path / "processed-cli"

    assert (
        main(
            [
                "preprocess",
                "amazon-reviews-2023",
                "--reviews-path",
                str(reviews_path),
                "--metadata-path",
                str(metadata_path),
                "--category",
                "all_beauty",
                "--output-dir",
                str(output),
                "--min-user-interactions",
                "3",
                "--min-item-interactions",
                "3",
                "--global-train-ratio",
                "0.5",
                "--global-val-ratio",
                "0.25",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert "processed Amazon Reviews 2023" in captured.out
    metadata = read_json(output / "metadata.json")
    assert metadata["num_interactions"] == 12
    assert metadata["timestamp_note"] == "Processed timestamp preserves the source integer value."
    assert metadata["raw_files"]["reviews"]["sha256"]
    assert metadata["raw_files"]["reviews"]["bytes"] > 0
    assert metadata["processed_file_checksums"]["interactions.csv"]["sha256"]
    assert read_json(output / "checksums.json")["files"]["config.yaml"]["bytes"] > 0
    assert "full-horizon k-core transductive" in metadata["global_time_protocol"]
    assert "average_rating" in metadata["excluded_metadata_columns"]
    assert (output / "command.txt").read_text(encoding="utf-8").startswith(
        "tglrec preprocess amazon-reviews-2023"
    )


def test_amazon_falls_back_to_asin_when_parent_asin_missing(tmp_path: Path):
    reviews_path = _write_jsonl_gz(
        tmp_path / "raw" / "reviews.jsonl.gz",
        [
            {"user_id": "u1", "asin": "child-2", "rating": 1, "timestamp": 20},
            {"user_id": "u1", "asin": "child-1", "rating": 1, "timestamp": 10},
            {"user_id": "u1", "asin": "child-3", "rating": 1, "timestamp": 30},
        ],
    )

    preprocess_amazon_reviews_2023(
        reviews_path=reviews_path,
        output_dir=tmp_path / "processed",
        min_user_interactions=3,
        min_item_interactions=1,
        global_train_ratio=0.4,
        global_val_ratio=0.3,
        seed=2026,
    )

    interactions = pd.read_csv(tmp_path / "processed" / "interactions.csv")
    assert list(interactions[schema.RAW_ITEM_ID]) == ["child-1", "child-2", "child-3"]


def test_amazon_collapses_duplicate_user_items_by_default(tmp_path: Path):
    reviews_path = _write_jsonl_gz(
        tmp_path / "raw" / "reviews.jsonl.gz",
        [
            {"user_id": "u1", "parent_asin": "A", "rating": 1, "timestamp": 10},
            {"user_id": "u1", "parent_asin": "A", "rating": 5, "timestamp": 15},
            {"user_id": "u1", "parent_asin": "B", "rating": 1, "timestamp": 20},
            {"user_id": "u1", "parent_asin": "C", "rating": 1, "timestamp": 30},
            {"user_id": "u2", "parent_asin": "A", "rating": 1, "timestamp": 11},
            {"user_id": "u2", "parent_asin": "B", "rating": 1, "timestamp": 21},
            {"user_id": "u2", "parent_asin": "C", "rating": 1, "timestamp": 31},
            {"user_id": "u3", "parent_asin": "A", "rating": 1, "timestamp": 12},
            {"user_id": "u3", "parent_asin": "B", "rating": 1, "timestamp": 22},
            {"user_id": "u3", "parent_asin": "C", "rating": 1, "timestamp": 32},
        ],
    )

    preprocess_amazon_reviews_2023(
        reviews_path=reviews_path,
        output_dir=tmp_path / "processed",
        min_user_interactions=3,
        min_item_interactions=3,
        global_train_ratio=0.5,
        global_val_ratio=0.25,
        seed=2026,
    )

    interactions = pd.read_csv(tmp_path / "processed" / "interactions.csv")
    metadata = read_json(tmp_path / "processed" / "metadata.json")
    assert len(interactions) == 9
    assert metadata["num_duplicate_user_item_rows_removed"] == 1
    assert not interactions.duplicated([schema.RAW_USER_ID, schema.RAW_ITEM_ID]).any()

    for _, user_events in interactions.groupby(schema.USER_ID):
        train_items = set(user_events.loc[user_events[schema.SPLIT_LOO] == "train", schema.ITEM_ID])
        heldout_items = set(
            user_events.loc[user_events[schema.SPLIT_LOO].isin(["val", "test"]), schema.ITEM_ID]
        )
        assert train_items.isdisjoint(heldout_items)


def test_amazon_rejects_same_user_timestamp_ties_by_default(tmp_path: Path):
    reviews_path = _write_jsonl_gz(
        tmp_path / "raw" / "reviews.jsonl.gz",
        [
            {"user_id": "u1", "parent_asin": "A", "rating": 1, "timestamp": 10},
            {"user_id": "u1", "parent_asin": "B", "rating": 1, "timestamp": 10},
            {"user_id": "u1", "parent_asin": "C", "rating": 1, "timestamp": 30},
        ],
    )

    with pytest.raises(ValueError, match="same-user events with identical timestamps"):
        preprocess_amazon_reviews_2023(
            reviews_path=reviews_path,
            output_dir=tmp_path / "processed",
            min_user_interactions=3,
            min_item_interactions=1,
            global_train_ratio=0.4,
            global_val_ratio=0.3,
            seed=2026,
        )


def test_amazon_allowed_same_timestamp_ties_are_recorded(tmp_path: Path):
    reviews_path = _write_jsonl_gz(
        tmp_path / "raw" / "reviews.jsonl.gz",
        [
            {"user_id": "u1", "parent_asin": "A", "rating": 1, "timestamp": 5},
            {"user_id": "u1", "parent_asin": "B", "rating": 1, "timestamp": 10},
            {"user_id": "u1", "parent_asin": "C", "rating": 1, "timestamp": 10},
            {"user_id": "u1", "parent_asin": "D", "rating": 1, "timestamp": 20},
            {"user_id": "u1", "parent_asin": "E", "rating": 1, "timestamp": 30},
        ],
    )

    preprocess_amazon_reviews_2023(
        reviews_path=reviews_path,
        output_dir=tmp_path / "processed",
        min_user_interactions=3,
        min_item_interactions=1,
        global_train_ratio=0.4,
        global_val_ratio=0.2,
        allow_same_timestamp_user_events=True,
        seed=2026,
    )

    metadata = read_json(tmp_path / "processed" / "metadata.json")
    assert metadata["same_user_timestamp_tie_stats_after_dedup"]["tied_extra_rows"] == 1
    assert metadata["same_user_timestamp_tie_stats"]["tied_extra_rows"] == 1
    assert metadata["processed_file_checksums"]["interactions.csv"]["sha256"]
    assert read_json(tmp_path / "processed" / "checksums.json")["files"]["metadata.json"]["sha256"]
