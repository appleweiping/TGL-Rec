"""Config-driven preprocessing for tiny JSONL datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.data.base import (
    REQUIRED_INTERACTION_FIELDS,
    REQUIRED_ITEM_FIELDS,
    DataSchemaError,
    PreprocessResult,
    require_fields,
)
from llm4rec.data.candidates import build_candidate_sets
from llm4rec.data.movielens_adapter import preprocess_movielens_from_config
from llm4rec.data.splits import build_user_histories, leave_one_out_split
from llm4rec.experiments.config import load_yaml_config, resolve_path
from llm4rec.io.artifacts import ensure_dir, read_jsonl, write_json, write_jsonl


def preprocess_from_config(config_or_path: dict[str, Any] | str | Path) -> PreprocessResult:
    """Run preprocessing from a dataset config path or mapping."""

    config = (
        load_yaml_config(config_or_path)
        if isinstance(config_or_path, (str, Path))
        else dict(config_or_path)
    )
    dataset = config.get("dataset", config)
    dataset_name = str(dataset.get("name", "unknown"))
    if dataset.get("adapter") == "movielens_style":
        return preprocess_movielens_from_config({"dataset": dataset})
    if dataset.get("adapter") != "tiny_jsonl":
        raise ValueError("Preprocessing supports adapter='tiny_jsonl' or 'movielens_style'.")

    paths = dataset.get("paths", {})
    interactions_path = resolve_path(paths["interactions"])
    items_path = resolve_path(paths["items"])
    output_dir = ensure_dir(resolve_path(dataset["output_dir"]))
    seed = int(dataset.get("seed", 0))
    split_strategy = str(dataset.get("split_strategy", "leave_one_out"))
    candidate_protocol = str(dataset.get("candidate_protocol", "full_catalog"))
    if split_strategy != "leave_one_out":
        raise ValueError("Phase 1 supports only split_strategy='leave_one_out'.")

    interactions = _load_interactions(interactions_path)
    items = _load_items(items_path)
    item_ids = [str(row["item_id"]) for row in items]
    missing_items = sorted({str(row["item_id"]) for row in interactions} - set(item_ids))
    if missing_items:
        raise DataSchemaError(f"Interactions reference missing item ids: {missing_items}")

    labeled = leave_one_out_split(interactions)
    histories = [
        {"user_id": user_id, "history": history}
        for user_id, history in sorted(build_user_histories(labeled).items())
    ]
    candidate_rows = build_candidate_sets(
        labeled,
        item_ids,
        protocol=candidate_protocol,
    )
    train_rows = [row for row in labeled if row["split"] == "train"]
    valid_rows = [row for row in labeled if row["split"] == "valid"]
    test_rows = [row for row in labeled if row["split"] == "test"]

    write_jsonl(output_dir / "interactions.jsonl", labeled)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "valid.jsonl", valid_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)
    write_jsonl(output_dir / "items.jsonl", sorted(items, key=lambda row: str(row["item_id"])))
    write_jsonl(output_dir / "histories.jsonl", histories)
    write_jsonl(output_dir / "candidates.jsonl", candidate_rows)
    metadata = {
        "candidate_protocol": candidate_protocol,
        "dataset": dataset_name,
        "interaction_count": len(interactions),
        "item_count": len(items),
        "seed": seed,
        "split_counts": {
            "train": len(train_rows),
            "valid": len(valid_rows),
            "test": len(test_rows),
        },
        "split_strategy": split_strategy,
        "user_count": len({str(row["user_id"]) for row in interactions}),
    }
    write_json(output_dir / "metadata.json", metadata)
    return PreprocessResult(output_dir=output_dir, metadata=metadata)


def _load_interactions(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for index, row in enumerate(rows):
        require_fields(row, REQUIRED_INTERACTION_FIELDS, label=f"interaction row {index}")
        row["user_id"] = str(row["user_id"])
        row["item_id"] = str(row["item_id"])
        if row["timestamp"] is not None and not isinstance(row["timestamp"], (int, float)):
            raise DataSchemaError(f"interaction row {index} has non-numeric timestamp")
        if row["rating"] is not None:
            row["rating"] = float(row["rating"])
        if row["domain"] is not None:
            row["domain"] = str(row["domain"])
    return rows


def _load_items(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    seen: set[str] = set()
    for index, row in enumerate(rows):
        require_fields(row, REQUIRED_ITEM_FIELDS, label=f"item row {index}")
        item_id = str(row["item_id"])
        if item_id in seen:
            raise DataSchemaError(f"duplicate item_id in item catalog: {item_id}")
        seen.add(item_id)
        row["item_id"] = item_id
        row["title"] = str(row["title"])
        if row["domain"] is not None:
            row["domain"] = str(row["domain"])
    return rows
