"""MovieLens-style adapter for llm4rec JSONL artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from llm4rec.data.base import DataSchemaError, PreprocessResult
from llm4rec.data.candidates import build_candidate_sets
from llm4rec.data.protocols import filter_min_interactions, subsample_interactions, temporal_split
from llm4rec.data.splits import build_user_histories, leave_one_out_split
from llm4rec.experiments.config import resolve_path
from llm4rec.io.artifacts import ensure_dir, read_jsonl, write_json, write_jsonl

MISSING_MOVIELENS_MESSAGE = (
    "MovieLens-style data is missing. Expected either processed CSV files at "
    "artifacts/datasets/movielens_1m/{interactions.csv,items.csv} or raw ML-1M files at "
    "data/raw/movielens_1m/ml-1m/{ratings.dat,movies.dat}. Run the existing tglrec MovieLens "
    "preprocess command or set dataset.paths.processed_dir / raw_dir in the config."
)


def preprocess_movielens_from_config(config: dict[str, Any]) -> PreprocessResult:
    """Read MovieLens-style data and save Phase 1-compatible JSONL artifacts."""

    dataset = dict(config.get("dataset", config))
    paths = dataset.get("paths", {})
    output_dir = ensure_dir(resolve_path(dataset["output_dir"]))
    split_strategy = str(dataset.get("split_strategy", "leave_one_out"))
    candidate_protocol = str(dataset.get("candidate_protocol", "full_catalog"))
    seed = int(dataset.get("seed", 0))
    min_user_interactions = int(dataset.get("min_user_interactions", 3))
    sample = dict(dataset.get("sample", {}))

    interactions, items = load_movielens_style(paths)
    interactions = filter_min_interactions(
        interactions,
        min_user_interactions=min_user_interactions,
    )
    interactions = subsample_interactions(
        interactions,
        max_users=_optional_int(sample.get("max_users")),
        max_items=_optional_int(sample.get("max_items")),
        max_interactions=_optional_int(sample.get("max_interactions")),
    )
    interactions = filter_min_interactions(
        interactions,
        min_user_interactions=min_user_interactions,
    )
    item_ids_in_data = {str(row["item_id"]) for row in interactions}
    items = [row for row in items if str(row["item_id"]) in item_ids_in_data]
    interactions, items = remap_user_item_ids(interactions, items)

    if split_strategy == "leave_one_out":
        labeled = leave_one_out_split(interactions)
    elif split_strategy == "temporal":
        labeled = temporal_split(
            interactions,
            train_ratio=float(dataset.get("train_ratio", 0.8)),
            valid_ratio=float(dataset.get("valid_ratio", 0.1)),
        )
    else:
        raise ValueError(f"Unsupported MovieLens split_strategy: {split_strategy}")
    item_ids = [str(row["item_id"]) for row in items]
    candidate_rows = build_candidate_sets(labeled, item_ids, protocol=candidate_protocol)
    histories = [
        {"user_id": user_id, "history": history}
        for user_id, history in sorted(build_user_histories(labeled).items())
    ]
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
        "adapter": "movielens_style",
        "candidate_protocol": candidate_protocol,
        "dataset": str(dataset.get("name", "movielens")),
        "interaction_count": len(interactions),
        "item_count": len(items),
        "missing_data_message": MISSING_MOVIELENS_MESSAGE,
        "sample": sample,
        "sampled": any(value not in (None, 0, "", False) for value in sample.values()),
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


def load_movielens_style(paths: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load processed CSV or raw ML-1M dat files."""

    processed_dir = paths.get("processed_dir")
    if processed_dir:
        processed = resolve_path(processed_dir)
        interactions_path = processed / "interactions.csv"
        items_path = processed / "items.csv"
        if interactions_path.is_file() and items_path.is_file():
            return _load_processed_csv(interactions_path, items_path)
    raw_dir = paths.get("raw_dir")
    if raw_dir:
        raw = resolve_path(raw_dir)
        ratings_path = raw / "ratings.dat"
        movies_path = raw / "movies.dat"
        if ratings_path.is_file() and movies_path.is_file():
            return _load_raw_dat(ratings_path, movies_path)
    jsonl_interactions = paths.get("interactions")
    jsonl_items = paths.get("items")
    if jsonl_interactions and jsonl_items:
        interactions_path = resolve_path(jsonl_interactions)
        items_path = resolve_path(jsonl_items)
        if interactions_path.is_file() and items_path.is_file():
            return read_jsonl(interactions_path), read_jsonl(items_path)
    raise FileNotFoundError(MISSING_MOVIELENS_MESSAGE)


def remap_user_item_ids(
    interactions: list[dict[str, Any]],
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Map raw ids to deterministic internal u*/i* ids."""

    user_raw_ids = sorted({str(row["user_id"]) for row in interactions}, key=_natural_key)
    item_raw_ids = sorted({str(row["item_id"]) for row in interactions}, key=_natural_key)
    user_map = {raw_id: f"u{index + 1}" for index, raw_id in enumerate(user_raw_ids)}
    item_map = {raw_id: f"i{index + 1}" for index, raw_id in enumerate(item_raw_ids)}
    item_by_raw = {str(row["item_id"]): row for row in items}
    remapped_interactions: list[dict[str, Any]] = []
    for row in interactions:
        raw_user = str(row["user_id"])
        raw_item = str(row["item_id"])
        if raw_item not in item_map:
            continue
        remapped_interactions.append(
            {
                "domain": row.get("domain", "movielens"),
                "item_id": item_map[raw_item],
                "rating": None if row.get("rating") is None else float(row["rating"]),
                "raw_item_id": raw_item,
                "raw_user_id": raw_user,
                "timestamp": None if row.get("timestamp") is None else float(row["timestamp"]),
                "user_id": user_map[raw_user],
            }
        )
    remapped_items: list[dict[str, Any]] = []
    for raw_item in item_raw_ids:
        row = item_by_raw.get(raw_item)
        if row is None:
            raise DataSchemaError(f"Missing MovieLens item metadata for item_id={raw_item}")
        title = str(row.get("title", raw_item))
        category = row.get("category", row.get("genres"))
        raw_text = row.get("raw_text") or " ".join(part for part in [title, str(category or "")] if part)
        remapped_items.append(
            {
                "brand": None,
                "category": None if category is None else str(category),
                "description": None,
                "domain": row.get("domain", "movielens"),
                "item_id": item_map[raw_item],
                "raw_item_id": raw_item,
                "raw_text": raw_text,
                "title": title,
            }
        )
    return remapped_interactions, remapped_items


def _load_processed_csv(
    interactions_path: Path,
    items_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    interactions: list[dict[str, Any]] = []
    with interactions_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            interactions.append(
                {
                    "domain": "movielens",
                    "item_id": str(row.get("item_id") or row.get("raw_item_id")),
                    "rating": _optional_float(row.get("rating")),
                    "timestamp": _optional_float(row.get("timestamp")),
                    "user_id": str(row.get("user_id") or row.get("raw_user_id")),
                }
            )
    items: list[dict[str, Any]] = []
    with items_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            title = str(row.get("title") or row.get("item_id"))
            genres = row.get("genres") or row.get("category")
            items.append(
                {
                    "brand": None,
                    "category": genres,
                    "description": None,
                    "domain": "movielens",
                    "item_id": str(row.get("item_id") or row.get("raw_item_id")),
                    "raw_text": " ".join(part for part in [title, str(genres or "")] if part),
                    "title": title,
                }
            )
    return interactions, items


def _load_raw_dat(
    ratings_path: Path,
    movies_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    interactions: list[dict[str, Any]] = []
    with ratings_path.open("r", encoding="latin-1") as handle:
        for line in handle:
            user_id, item_id, rating, timestamp = line.rstrip("\n").split("::")
            interactions.append(
                {
                    "domain": "movielens",
                    "item_id": item_id,
                    "rating": float(rating),
                    "timestamp": float(timestamp),
                    "user_id": user_id,
                }
            )
    items: list[dict[str, Any]] = []
    with movies_path.open("r", encoding="latin-1") as handle:
        for line in handle:
            item_id, title, genres = line.rstrip("\n").split("::")
            items.append(
                {
                    "brand": None,
                    "category": genres,
                    "description": None,
                    "domain": "movielens",
                    "item_id": item_id,
                    "raw_text": f"{title} {genres}",
                    "title": title,
                }
            )
    return interactions, items


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value in (None, "", 0):
        return None
    return int(value)


def _natural_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):020d}")
    except ValueError:
        return (1, value)
