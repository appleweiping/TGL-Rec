"""Amazon Reviews 2023 local-file preprocessing.

Source pages checked 2026-04-29:
- https://amazon-reviews-2023.github.io/main.html
- https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tglrec.data import schema
from tglrec.data.artifacts import (
    CHECKSUM_MANIFEST_NAME,
    COMMON_ARTIFACT_MANIFEST_FILES,
    PROCESSED_DATASET_FILES,
    build_checksum_manifest,
    file_fingerprint,
    same_user_timestamp_tie_stats,
    write_checksum_manifest,
)
from tglrec.data.splits import (
    apply_stable_ids,
    assert_no_future_leakage,
    assign_event_ids,
    global_time_split,
    iterative_min_filter,
    temporal_leave_one_out_split,
)
from tglrec.utils.io import ensure_dir
from tglrec.utils.logging import write_artifact_manifest

AMAZON_REVIEWS_2023_SOURCE_PAGE = "https://amazon-reviews-2023.github.io/main.html"
AMAZON_REVIEWS_2023_HF = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023"
DEFAULT_ITEM_TEXT_COLUMNS = [
    "title",
    "main_category",
    "categories",
    "store",
    "description",
    "features",
]
EXCLUDED_AGGREGATE_METADATA_COLUMNS = ["average_rating", "rating_number"]


@dataclass(frozen=True)
class PreprocessResult:
    """Summary of a completed preprocessing run."""

    output_dir: Path
    num_interactions: int
    num_users: int
    num_items: int
    metadata: dict[str, Any]


def preprocess_amazon_reviews_2023(
    *,
    reviews_path: str | Path,
    output_dir: str | Path,
    metadata_path: str | Path | None = None,
    category: str | None = None,
    user_col: str = "user_id",
    item_col: str = "parent_asin",
    item_fallback_col: str = "asin",
    timestamp_col: str | None = None,
    rating_col: str = "rating",
    metadata_item_col: str = "parent_asin",
    min_rating: float | None = None,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    global_train_ratio: float = 0.8,
    global_val_ratio: float = 0.1,
    deduplicate_user_items: bool = True,
    allow_same_timestamp_user_events: bool = False,
    source_file_url: str | None = None,
    metadata_source_url: str | None = None,
    hf_revision: str | None = None,
    seed: int = 2026,
    command: str | None = None,
) -> PreprocessResult:
    """Preprocess a local Amazon Reviews 2023 category file into temporal splits."""

    reviews_file = Path(reviews_path)
    raw_reviews = _read_local_table(reviews_file, "reviews")
    raw_review_rows = len(raw_reviews)
    source_timestamp_col = _resolve_timestamp_col(raw_reviews, timestamp_col)
    interactions = _normalize_reviews(
        raw_reviews,
        user_col=user_col,
        item_col=item_col,
        item_fallback_col=item_fallback_col,
        timestamp_col=source_timestamp_col,
        rating_col=rating_col,
        min_rating=min_rating,
    )
    rows_after_rating_filter = len(interactions)
    duplicate_user_item_rows_removed = 0
    if deduplicate_user_items:
        before_dedup = len(interactions)
        interactions = _deduplicate_user_items(interactions)
        duplicate_user_item_rows_removed = before_dedup - len(interactions)

    same_user_timestamp_tie_stats_after_dedup = same_user_timestamp_tie_stats(interactions)
    if (
        same_user_timestamp_tie_stats_after_dedup["tied_extra_rows"] > 0
        and not allow_same_timestamp_user_events
    ):
        raise ValueError(
            "Amazon Reviews 2023 contains same-user events with identical timestamps after "
            "normalization "
            f"({same_user_timestamp_tie_stats_after_dedup['tied_extra_rows']} tied extra rows). "
            "This makes temporal "
            "leave-one-out order ambiguous. Use --allow-same-timestamp-user-events only for "
            "exploratory preprocessing after documenting the policy."
        )

    filtered = iterative_min_filter(
        interactions,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )
    mapped, users, item_ids = apply_stable_ids(filtered)

    item_metadata = _load_item_metadata(
        metadata_path=Path(metadata_path) if metadata_path is not None else None,
        metadata_item_col=metadata_item_col,
    )
    items = item_ids.merge(item_metadata, on=schema.RAW_ITEM_ID, how="left", validate="one_to_one")
    for column in DEFAULT_ITEM_TEXT_COLUMNS:
        if column not in items.columns:
            items[column] = ""
        items[column] = items[column].map(_stable_text_value)

    split_ready = assign_event_ids(mapped)
    split_ready[schema.SPLIT_LOO] = temporal_leave_one_out_split(split_ready)
    split_ready[schema.SPLIT_GLOBAL], cutoffs = global_time_split(
        split_ready, train_ratio=global_train_ratio, val_ratio=global_val_ratio
    )
    interactions_out = split_ready[schema.INTERACTION_COLUMNS].copy()
    assert_no_future_leakage(interactions_out, schema.SPLIT_LOO)
    assert_no_future_leakage(interactions_out, schema.SPLIT_GLOBAL)

    root = ensure_dir(output_dir)
    interactions_out.to_csv(root / "interactions.csv", index=False)
    users.to_csv(root / "users.csv", index=False)
    items[[schema.ITEM_ID, schema.RAW_ITEM_ID, *DEFAULT_ITEM_TEXT_COLUMNS]].to_csv(
        root / "items.csv", index=False
    )
    _write_split_files(interactions_out, root, schema.SPLIT_LOO, "temporal_leave_one_out")
    _write_split_files(interactions_out, root, schema.SPLIT_GLOBAL, "global_time")
    processed_files = PROCESSED_DATASET_FILES
    processed_file_checksums = build_checksum_manifest(root, processed_files)["files"]

    dataset_name = "amazon_reviews_2023" if category is None else f"amazon_reviews_2023_{category}"
    config = {
        "dataset": dataset_name,
        "source_page": AMAZON_REVIEWS_2023_SOURCE_PAGE,
        "huggingface": AMAZON_REVIEWS_2023_HF,
        "reviews_path": str(reviews_file),
        "metadata_path": str(metadata_path) if metadata_path is not None else None,
        "category": category,
        "user_col": user_col,
        "item_col": item_col,
        "item_fallback_col": item_fallback_col,
        "timestamp_col": source_timestamp_col,
        "rating_col": rating_col,
        "metadata_item_col": metadata_item_col,
        "min_rating": min_rating,
        "min_user_interactions": min_user_interactions,
        "min_item_interactions": min_item_interactions,
        "global_train_ratio": global_train_ratio,
        "global_val_ratio": global_val_ratio,
        "deduplicate_user_items": deduplicate_user_items,
        "allow_same_timestamp_user_events": allow_same_timestamp_user_events,
        "source_file_url": source_file_url,
        "metadata_source_url": metadata_source_url,
        "hf_revision": hf_revision,
        "seed": seed,
    }
    metadata: dict[str, Any] = {
        "dataset": dataset_name,
        "source_page": AMAZON_REVIEWS_2023_SOURCE_PAGE,
        "huggingface": AMAZON_REVIEWS_2023_HF,
        "date_checked": "2026-04-29",
        "reviews_path": str(reviews_file),
        "metadata_path": str(metadata_path) if metadata_path is not None else None,
        "raw_files": {
            "reviews": file_fingerprint(reviews_file, include_path=True),
            "metadata": (
                file_fingerprint(Path(metadata_path), include_path=True)
                if metadata_path is not None
                else None
            ),
        },
        "source_file_url": source_file_url,
        "metadata_source_url": metadata_source_url,
        "hf_revision": hf_revision,
        "license_note": (
            "Verify the current source page and Hugging Face dataset card before "
            "redistributing raw or processed Amazon Reviews 2023 artifacts."
        ),
        "category": category,
        "num_raw_review_rows": int(raw_review_rows),
        "num_rows_after_rating_filter": int(rows_after_rating_filter),
        "num_duplicate_user_item_rows_removed": int(duplicate_user_item_rows_removed),
        "num_interactions": int(len(interactions_out)),
        "num_users": int(users[schema.USER_ID].nunique()),
        "num_items": int(items[schema.ITEM_ID].nunique()),
        "min_timestamp": int(interactions_out[schema.TIMESTAMP].min()),
        "max_timestamp": int(interactions_out[schema.TIMESTAMP].max()),
        "timestamp_note": "Processed timestamp preserves the source integer value.",
        "same_user_timestamp_tie_stats_after_dedup": same_user_timestamp_tie_stats_after_dedup,
        "same_user_timestamp_tie_stats": same_user_timestamp_tie_stats(interactions_out),
        "temporal_loo_tie_policy": (
            "same-user identical timestamps are rejected by default; if explicitly allowed, "
            "ties are broken deterministically by item/event id and are not paper-grade."
        ),
        "global_time_protocol": (
            "full-horizon k-core transductive split; do not use for inductive claims until "
            "train-period-only filtering and ID mapping are implemented."
        ),
        "excluded_metadata_columns": EXCLUDED_AGGREGATE_METADATA_COLUMNS,
        "split_summary": {
            "temporal_leave_one_out": _split_summary(interactions_out, schema.SPLIT_LOO),
            "global_time": _split_summary(interactions_out, schema.SPLIT_GLOBAL),
        },
        "processed_file_checksums": processed_file_checksums,
        "global_time_cutoffs": {
            "train_end_exclusive": cutoffs.train_end,
            "val_end_exclusive": cutoffs.val_end,
        },
        "files": {
            "interactions": "interactions.csv",
            "users": "users.csv",
            "items": "items.csv",
            "temporal_leave_one_out": "temporal_leave_one_out/{train,val,test}.csv",
            "global_time": "global_time/{train,val,test}.csv",
            "checksums": CHECKSUM_MANIFEST_NAME,
        },
    }
    manifest_command = command or (" ".join(sys.argv) if sys.argv else "tglrec preprocess amazon-reviews-2023")
    write_artifact_manifest(root, command=manifest_command, config=config, metadata=metadata)
    write_checksum_manifest(root, [*processed_files, *COMMON_ARTIFACT_MANIFEST_FILES])
    return PreprocessResult(
        output_dir=root,
        num_interactions=len(interactions_out),
        num_users=metadata["num_users"],
        num_items=metadata["num_items"],
        metadata=metadata,
    )


def _read_local_table(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Amazon Reviews 2023 {label} file: {path}")
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes[-2:] in (
        [".jsonl", ".gz"],
        [".json", ".gz"],
        [".ndjson", ".gz"],
    ) or suffixes[-1:] in ([".jsonl"], [".ndjson"]):
        return pd.read_json(path, lines=True, compression="infer")
    if suffixes[-1:] == [".csv"]:
        return pd.read_csv(path)
    raise ValueError(
        f"Unsupported Amazon Reviews 2023 {label} format: {path}. "
        "Use .jsonl, .jsonl.gz, .json.gz, .ndjson, or .csv."
    )


def _resolve_timestamp_col(reviews: pd.DataFrame, timestamp_col: str | None) -> str:
    if timestamp_col is not None:
        if timestamp_col not in reviews.columns:
            raise ValueError(f"Missing requested timestamp column: {timestamp_col}")
        return timestamp_col
    for candidate in ("timestamp", "unixReviewTime"):
        if candidate in reviews.columns:
            return candidate
    raise ValueError("Could not find a timestamp column. Pass --timestamp-col explicitly.")


def _normalize_reviews(
    reviews: pd.DataFrame,
    *,
    user_col: str,
    item_col: str,
    item_fallback_col: str,
    timestamp_col: str,
    rating_col: str,
    min_rating: float | None,
) -> pd.DataFrame:
    _require_columns(reviews, [user_col, timestamp_col], label="reviews")
    raw_item_ids = _coalesced_item_ids(reviews, item_col=item_col, fallback_col=item_fallback_col)
    if rating_col in reviews.columns:
        ratings = pd.to_numeric(reviews[rating_col], errors="raise").astype(float)
    else:
        ratings = pd.Series(1.0, index=reviews.index, dtype="float64")

    normalized = pd.DataFrame(
        {
            schema.RAW_USER_ID: reviews[user_col].map(_stable_text_value),
            schema.RAW_ITEM_ID: raw_item_ids.map(_stable_text_value),
            schema.TIMESTAMP: pd.to_numeric(reviews[timestamp_col], errors="raise").astype("int64"),
            schema.RATING: ratings,
            "_source_row": range(len(reviews)),
        }
    )
    if min_rating is not None:
        normalized = normalized.loc[normalized[schema.RATING] >= min_rating].copy()
    invalid = (
        normalized[schema.RAW_USER_ID].eq("")
        | normalized[schema.RAW_ITEM_ID].eq("")
        | normalized[schema.TIMESTAMP].isna()
    )
    if invalid.any():
        raise ValueError(
            "Amazon Reviews 2023 reviews contain missing user, item, or timestamp values: "
            f"{int(invalid.sum())} invalid rows."
        )
    return normalized


def _deduplicate_user_items(interactions: pd.DataFrame) -> pd.DataFrame:
    ordered = interactions.sort_values(
        [
            schema.RAW_USER_ID,
            schema.RAW_ITEM_ID,
            schema.TIMESTAMP,
            "_source_row",
        ],
        kind="mergesort",
    )
    deduped = ordered.drop_duplicates(
        subset=[schema.RAW_USER_ID, schema.RAW_ITEM_ID],
        keep="first",
    )
    return deduped.drop(columns=["_source_row"]).reset_index(drop=True)


def _coalesced_item_ids(reviews: pd.DataFrame, *, item_col: str, fallback_col: str) -> pd.Series:
    if item_col not in reviews.columns and fallback_col not in reviews.columns:
        raise ValueError(f"Missing item columns: expected {item_col} or {fallback_col}")
    if item_col in reviews.columns and fallback_col in reviews.columns:
        primary = reviews[item_col].where(reviews[item_col].notna(), reviews[fallback_col])
        return primary.where(primary.astype(str).str.len() > 0, reviews[fallback_col])
    if item_col in reviews.columns:
        return reviews[item_col]
    return reviews[fallback_col]


def _load_item_metadata(metadata_path: Path | None, *, metadata_item_col: str) -> pd.DataFrame:
    if metadata_path is None:
        return pd.DataFrame(columns=[schema.RAW_ITEM_ID, *DEFAULT_ITEM_TEXT_COLUMNS])
    metadata = _read_local_table(metadata_path, "metadata")
    _require_columns(metadata, [metadata_item_col], label="metadata")
    metadata = metadata.copy()
    metadata[schema.RAW_ITEM_ID] = metadata[metadata_item_col].map(_stable_text_value)
    metadata = metadata.loc[metadata[schema.RAW_ITEM_ID] != ""].copy()
    for column in DEFAULT_ITEM_TEXT_COLUMNS:
        if column not in metadata.columns:
            metadata[column] = ""
    metadata["_source_row"] = range(len(metadata))
    metadata = metadata.sort_values([schema.RAW_ITEM_ID, "_source_row"], kind="mergesort")
    metadata = metadata.drop_duplicates(subset=[schema.RAW_ITEM_ID], keep="first")
    return metadata[[schema.RAW_ITEM_ID, *DEFAULT_ITEM_TEXT_COLUMNS]]


def _require_columns(frame: pd.DataFrame, columns: list[str], *, label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing Amazon Reviews 2023 {label} columns: {missing}")


def _stable_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    if pd.isna(value):
        return ""
    return str(value)


def _write_split_files(interactions: pd.DataFrame, root: Path, split_col: str, split_name: str) -> None:
    split_dir = ensure_dir(root / split_name)
    for split in ("train", "val", "test"):
        frame = interactions.loc[interactions[split_col] == split].drop(
            columns=[schema.SPLIT_LOO, schema.SPLIT_GLOBAL]
        )
        frame.to_csv(split_dir / f"{split}.csv", index=False)


def _split_summary(interactions: pd.DataFrame, split_col: str) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        frame = interactions.loc[interactions[split_col] == split]
        summary[split] = {
            "rows": int(len(frame)),
            "users": int(frame[schema.USER_ID].nunique()),
            "items": int(frame[schema.ITEM_ID].nunique()),
        }
    return summary

