"""Export processed TGLRec datasets to RecBole atomic benchmark files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from tglrec.data import schema
from tglrec.data.artifacts import write_checksum_manifest
from tglrec.data.splits import assert_no_future_leakage
from tglrec.utils.config import write_config
from tglrec.utils.io import ensure_dir, write_json
from tglrec.utils.logging import current_git_commit


RECBLE_ATOMIC_FILES_URL = "https://recbole.io/atomic_files.html"
RECBLE_DATA_SETTINGS_URL = "https://recbole.io/docs/user_guide/config/data_settings.html"
RECBLE_EVALUATION_SETTINGS_URL = (
    "https://recbole.io/docs/user_guide/config/evaluation_settings.html"
)


@dataclass(frozen=True)
class RecBoleExportResult:
    """Summary of a completed RecBole export."""

    output_dir: Path
    dataset_name: str
    split_name: str
    num_train: int
    num_valid: int
    num_test: int


def export_recbole_general_cf(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path | None = None,
    dataset_name: str | None = None,
    split_name: str = "temporal_leave_one_out",
    command: str = "tglrec export recbole-general",
) -> RecBoleExportResult:
    """Write RecBole benchmark files for train/valid/test general-CF baselines."""

    dataset_root = Path(dataset_dir)
    interactions_path = dataset_root / "interactions.csv"
    users_path = dataset_root / "users.csv"
    items_path = dataset_root / "items.csv"
    if not interactions_path.exists():
        raise FileNotFoundError(f"Missing processed interactions: {interactions_path}")
    if not users_path.exists():
        raise FileNotFoundError(f"Missing processed users: {users_path}")
    if not items_path.exists():
        raise FileNotFoundError(f"Missing processed items: {items_path}")

    interactions = pd.read_csv(interactions_path)
    users = pd.read_csv(users_path)
    items = pd.read_csv(items_path)
    split_col = _split_column(split_name)
    _validate_export_inputs(interactions, users, items, split_col)

    export_name = dataset_name or _default_dataset_name(dataset_root, split_name)
    run_root = Path(output_dir) if output_dir is not None else Path("artifacts") / "recbole" / export_name
    data_root = ensure_dir(run_root / export_name)

    split_frames = {
        "train": _ordered_interactions(interactions.loc[interactions[split_col] == "train"]),
        "valid": _ordered_interactions(interactions.loc[interactions[split_col] == "val"]),
        "test": _ordered_interactions(interactions.loc[interactions[split_col] == "test"]),
    }
    for suffix, frame in split_frames.items():
        _write_inter_file(data_root / f"{export_name}.{suffix}.inter", frame)
    _write_user_file(data_root / f"{export_name}.user", users)
    _write_item_file(data_root / f"{export_name}.item", items)

    config = _recbole_general_config(run_root, export_name)
    write_config(config, run_root / "recbole_general_cf.yaml")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_root),
        "dataset_name": export_name,
        "export_type": "recbole_general_cf_benchmark",
        "num_test": len(split_frames["test"]),
        "num_train": len(split_frames["train"]),
        "num_users": int(users[schema.USER_ID].nunique()),
        "num_valid": len(split_frames["valid"]),
        "num_items": int(items[schema.ITEM_ID].nunique()),
        "recbole_docs_checked": {
            "access_date": "2026-05-01",
            "atomic_files": RECBLE_ATOMIC_FILES_URL,
            "benchmark_filename": RECBLE_DATA_SETTINGS_URL,
            "evaluation_settings": RECBLE_EVALUATION_SETTINGS_URL,
        },
        "split_column": split_col,
        "split_name": split_name,
        "warning": (
            "This export preserves project train/valid/test labels for general collaborative "
            "filtering models such as LightGCN or BPR. Sequential RecBole models need a separate "
            "history-aware export/adapter before reportable SASRec, BERT4Rec, or TiSASRec runs."
        ),
    }
    write_json(metadata, run_root / "metadata.json")
    (run_root / "command.txt").write_text(command + "\n", encoding="utf-8", newline="\n")
    (run_root / "git_commit.txt").write_text(
        current_git_commit(".") + "\n", encoding="utf-8", newline="\n"
    )
    _write_readme(run_root / "README.md", export_name)
    write_checksum_manifest(run_root, _manifest_paths(export_name))
    return RecBoleExportResult(
        output_dir=run_root,
        dataset_name=export_name,
        split_name=split_name,
        num_train=len(split_frames["train"]),
        num_valid=len(split_frames["valid"]),
        num_test=len(split_frames["test"]),
    )


def _split_column(split_name: str) -> str:
    if split_name == "temporal_leave_one_out":
        return schema.SPLIT_LOO
    if split_name == "global_time":
        return schema.SPLIT_GLOBAL
    raise ValueError("split_name must be 'temporal_leave_one_out' or 'global_time'")


def _validate_export_inputs(
    interactions: pd.DataFrame,
    users: pd.DataFrame,
    items: pd.DataFrame,
    split_col: str,
) -> None:
    missing = [column for column in schema.INTERACTION_COLUMNS if column not in interactions.columns]
    if missing:
        raise ValueError(f"interactions.csv is missing required columns: {missing}")
    if schema.USER_ID not in users.columns or schema.RAW_USER_ID not in users.columns:
        raise ValueError("users.csv must contain user_id and raw_user_id")
    if schema.ITEM_ID not in items.columns or schema.RAW_ITEM_ID not in items.columns:
        raise ValueError("items.csv must contain item_id and raw_item_id")
    assert_no_future_leakage(interactions, split_col)
    labels = set(str(label) for label in interactions[split_col].unique())
    if labels != {"train", "val", "test"}:
        raise ValueError(f"{split_col} must contain exactly train/val/test labels, got {labels}")
    interaction_users = set(interactions[schema.USER_ID].astype(int))
    known_users = set(users[schema.USER_ID].astype(int))
    if not interaction_users.issubset(known_users):
        raise ValueError("users.csv is missing user ids present in interactions.csv")
    interaction_items = set(interactions[schema.ITEM_ID].astype(int))
    known_items = set(items[schema.ITEM_ID].astype(int))
    if not interaction_items.issubset(known_items):
        raise ValueError("items.csv is missing item ids present in interactions.csv")


def _ordered_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        [schema.USER_ID, schema.TIMESTAMP, schema.ITEM_ID, schema.EVENT_ID],
        kind="mergesort",
    )


def _write_inter_file(path: Path, frame: pd.DataFrame) -> None:
    columns = [
        "user_id:token",
        "item_id:token",
        "rating:float",
        "timestamp:float",
        "event_id:token",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        for row in frame.itertuples(index=False):
            writer.writerow(
                [
                    int(getattr(row, schema.USER_ID)),
                    int(getattr(row, schema.ITEM_ID)),
                    float(getattr(row, schema.RATING)),
                    float(getattr(row, schema.TIMESTAMP)),
                    int(getattr(row, schema.EVENT_ID)),
                ]
            )


def _write_user_file(path: Path, users: pd.DataFrame) -> None:
    ordered = users.sort_values(schema.USER_ID, kind="mergesort")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["user_id:token", "raw_user_id:token"])
        for row in ordered.itertuples(index=False):
            writer.writerow([int(getattr(row, schema.USER_ID)), _sanitize_token(getattr(row, schema.RAW_USER_ID))])


def _write_item_file(path: Path, items: pd.DataFrame) -> None:
    ordered = items.sort_values(schema.ITEM_ID, kind="mergesort")
    has_title = "title" in ordered.columns
    header = ["item_id:token", "raw_item_id:token"]
    if has_title:
        header.append("title:token_seq")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for row in ordered.itertuples(index=False):
            values = [int(getattr(row, schema.ITEM_ID)), _sanitize_token(getattr(row, schema.RAW_ITEM_ID))]
            if has_title:
                values.append(_sanitize_token_sequence(getattr(row, "title")))
            writer.writerow(values)


def _sanitize_token(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    return text.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def _sanitize_token_sequence(value: Any) -> str:
    text = _sanitize_token(value)
    return " ".join(part for part in text.split(" ") if part)


def _default_dataset_name(dataset_root: Path, split_name: str) -> str:
    stem = dataset_root.name.lower().replace("-", "_")
    suffix = "loo" if split_name == "temporal_leave_one_out" else "global_time"
    return f"{stem}_{suffix}"


def _recbole_general_config(root: Path, dataset_name: str) -> dict[str, Any]:
    return {
        "ITEM_ID_FIELD": "item_id",
        "LABEL_FIELD": "rating",
        "RATING_FIELD": "rating",
        "TIME_FIELD": "timestamp",
        "USER_ID_FIELD": "user_id",
        "benchmark_filename": ["train", "valid", "test"],
        "data_path": str(root.resolve()),
        "dataset": dataset_name,
        "eval_args": {
            "group_by": "user",
            "mode": "full",
            "order": "TO",
            "split": {"RS": [0.8, 0.1, 0.1]},
        },
        "field_separator": "\\t",
        "load_col": {"inter": ["user_id", "item_id", "rating", "timestamp"]},
        "metric_decimal_place": 6,
        "metrics": ["Recall", "MRR", "NDCG", "Hit"],
        "normalize_all": False,
        "topk": [5, 10, 20],
        "valid_metric": "NDCG@10",
    }


def _write_readme(path: Path, dataset_name: str) -> None:
    path.write_text(
        "\n".join(
            [
                f"# RecBole General-CF Export: {dataset_name}",
                "",
                "This artifact contains RecBole atomic benchmark files for train/valid/test",
                "general collaborative-filtering baselines. Use the generated",
                "`recbole_general_cf.yaml` with RecBole models such as BPR or LightGCN.",
                "",
                "Do not use this artifact as-is for sequential RecBole models. SASRec,",
                "BERT4Rec, and TiSASRec need a history-aware adapter/export so validation",
                "and test targets are evaluated with the correct prior user histories.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )


def _manifest_paths(dataset_name: str) -> list[str]:
    return [
        "README.md",
        "command.txt",
        "git_commit.txt",
        "metadata.json",
        "recbole_general_cf.yaml",
        f"{dataset_name}/{dataset_name}.item",
        f"{dataset_name}/{dataset_name}.test.inter",
        f"{dataset_name}/{dataset_name}.train.inter",
        f"{dataset_name}/{dataset_name}.user",
        f"{dataset_name}/{dataset_name}.valid.inter",
    ]
