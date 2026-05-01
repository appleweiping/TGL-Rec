"""Leakage-safe sequence example export for sequential baselines and LLM prompts."""

from __future__ import annotations

import csv
import json
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


@dataclass(frozen=True)
class SequenceCaseExportResult:
    """Summary of a completed sequential case export."""

    output_dir: Path
    dataset_name: str
    split_name: str
    num_train_examples: int
    num_train_transitions_available: int
    num_validation_cases: int
    num_test_cases: int


@dataclass(frozen=True)
class _HistoryEvent:
    event_id: int
    item_id: int
    timestamp: int


def export_sequence_cases(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path | None = None,
    dataset_name: str | None = None,
    split_name: str = "temporal_leave_one_out",
    use_validation_history_for_test: bool = True,
    max_history_items: int = 50,
    write_train_examples: bool = False,
    command: str = "tglrec export sequence-cases",
) -> SequenceCaseExportResult:
    """Export train examples and valid/test cases with explicit prior histories."""

    if max_history_items < 0:
        raise ValueError("max_history_items must be non-negative; use 0 for full histories")
    dataset_root = Path(dataset_dir)
    interactions_path = dataset_root / "interactions.csv"
    if not interactions_path.exists():
        raise FileNotFoundError(f"Missing processed interactions: {interactions_path}")
    interactions = pd.read_csv(interactions_path)
    split_col = _split_column(split_name)
    _validate_interactions(interactions, split_col)

    export_name = dataset_name or _default_dataset_name(dataset_root, split_name)
    run_root = ensure_dir(
        Path(output_dir) if output_dir is not None else Path("artifacts") / "sequences" / export_name
    )
    train_rows, eval_rows, sequence_rows, train_transitions_available = _build_examples(
        interactions,
        split_col=split_col,
        use_validation_history_for_test=use_validation_history_for_test,
        max_history_items=max_history_items,
        write_train_examples=write_train_examples,
    )
    _write_rows(run_root / "train_examples.csv", train_rows, id_field="example_id")
    _write_rows(run_root / "eval_cases.csv", eval_rows, id_field="case_id")
    _write_user_sequences(run_root / "user_sequences.csv", sequence_rows)
    config = {
        "dataset_dir": str(dataset_root),
        "dataset_name": export_name,
        "export_type": "sequence_cases",
        "history_policy": {
            "max_history_items": max_history_items,
            "max_history_items_semantics": "0 means full prior history; otherwise keep the most recent N events",
            "train_examples": "recent prior train events from the same user",
            "validation_cases": "recent prior train events from the same user",
            "test_cases": (
                "recent prior train plus validation events from the same user"
                if use_validation_history_for_test
                else "recent prior train events from the same user"
            ),
        },
        "split_column": split_col,
        "split_name": split_name,
        "train_examples_materialized": write_train_examples,
        "use_validation_history_for_test": use_validation_history_for_test,
    }
    write_config(config, run_root / "config.yaml")
    write_json(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_dir": str(dataset_root),
            "dataset_name": export_name,
            "export_type": "sequence_cases",
            "num_test_cases": sum(row["target_split"] == "test" for row in eval_rows),
            "num_train_examples": len(train_rows),
            "num_train_transitions_available": train_transitions_available,
            "num_validation_cases": sum(row["target_split"] == "val" for row in eval_rows),
            "max_history_items": max_history_items,
            "split_column": split_col,
            "split_name": split_name,
            "train_examples_materialized": write_train_examples,
            "use_validation_history_for_test": use_validation_history_for_test,
        },
        run_root / "metadata.json",
    )
    (run_root / "command.txt").write_text(command + "\n", encoding="utf-8", newline="\n")
    (run_root / "git_commit.txt").write_text(
        current_git_commit(".") + "\n", encoding="utf-8", newline="\n"
    )
    _write_readme(run_root / "README.md")
    write_checksum_manifest(
        run_root,
        [
            "README.md",
            "command.txt",
            "config.yaml",
            "eval_cases.csv",
            "git_commit.txt",
            "metadata.json",
            "train_examples.csv",
            "user_sequences.csv",
        ],
    )
    return SequenceCaseExportResult(
        output_dir=run_root,
        dataset_name=export_name,
        split_name=split_name,
        num_train_examples=len(train_rows),
        num_train_transitions_available=train_transitions_available,
        num_validation_cases=sum(row["target_split"] == "val" for row in eval_rows),
        num_test_cases=sum(row["target_split"] == "test" for row in eval_rows),
    )


def _split_column(split_name: str) -> str:
    if split_name == "temporal_leave_one_out":
        return schema.SPLIT_LOO
    if split_name == "global_time":
        return schema.SPLIT_GLOBAL
    raise ValueError("split_name must be 'temporal_leave_one_out' or 'global_time'")


def _validate_interactions(interactions: pd.DataFrame, split_col: str) -> None:
    missing = [column for column in schema.INTERACTION_COLUMNS if column not in interactions.columns]
    if missing:
        raise ValueError(f"interactions.csv is missing required columns: {missing}")
    assert_no_future_leakage(interactions, split_col)
    labels = set(str(label) for label in interactions[split_col].unique())
    if not {"train", "val", "test"}.issubset(labels):
        raise ValueError(f"{split_col} must contain train, val, and test labels, got {labels}")


def _build_examples(
    interactions: pd.DataFrame,
    *,
    split_col: str,
    use_validation_history_for_test: bool,
    max_history_items: int,
    write_train_examples: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    example_id = 0
    train_transitions_available = 0
    for user_id, group in interactions.groupby(schema.USER_ID, sort=True):
        train_history: list[_HistoryEvent] = []
        train_val_history: list[_HistoryEvent] = []
        train_events: list[_HistoryEvent] = []
        val_events: list[_HistoryEvent] = []
        test_events: list[_HistoryEvent] = []
        ordered = group.sort_values(
            [schema.TIMESTAMP, schema.ITEM_ID, schema.EVENT_ID],
            kind="mergesort",
        )
        for row in ordered.itertuples(index=False):
            target = _history_event(row)
            split_label = str(getattr(row, split_col))
            if split_label == "train":
                if train_history:
                    train_transitions_available += 1
                    if write_train_examples:
                        train_rows.append(
                            _example_row(
                                row_id=example_id,
                                id_field="example_id",
                                user_id=int(user_id),
                                target=target,
                                target_split="train",
                                history=_history_tail(train_history, max_history_items),
                            )
                        )
                        example_id += 1
                train_history.append(target)
                train_val_history.append(target)
                train_events.append(target)
            elif split_label == "val":
                eval_rows.append(
                    _example_row(
                        row_id=target.event_id,
                        id_field="case_id",
                        user_id=int(user_id),
                        target=target,
                        target_split="val",
                        history=_history_tail(train_history, max_history_items),
                    )
                )
                if use_validation_history_for_test:
                    train_val_history.append(target)
                val_events.append(target)
            elif split_label == "test":
                eval_rows.append(
                    _example_row(
                        row_id=target.event_id,
                        id_field="case_id",
                        user_id=int(user_id),
                        target=target,
                        target_split="test",
                        history=_history_tail(train_val_history, max_history_items),
                    )
                )
                test_events.append(target)
            else:
                raise ValueError(f"Unexpected split label {split_label!r}")
        sequence_rows.append(
            _sequence_row(
                user_id=int(user_id),
                train_events=train_events,
                val_events=val_events,
                test_events=test_events,
            )
        )
    return (
        train_rows,
        sorted(eval_rows, key=lambda row: (row["target_timestamp"], row["case_id"])),
        sequence_rows,
        train_transitions_available,
    )


def _history_tail(history: list[_HistoryEvent], max_history_items: int) -> list[_HistoryEvent]:
    if max_history_items == 0 or len(history) <= max_history_items:
        return list(history)
    return history[-max_history_items:]


def _history_event(row: Any) -> _HistoryEvent:
    return _HistoryEvent(
        event_id=int(getattr(row, schema.EVENT_ID)),
        item_id=int(getattr(row, schema.ITEM_ID)),
        timestamp=int(getattr(row, schema.TIMESTAMP)),
    )


def _example_row(
    *,
    row_id: int,
    id_field: str,
    user_id: int,
    target: _HistoryEvent,
    target_split: str,
    history: list[_HistoryEvent],
) -> dict[str, Any]:
    return {
        id_field: row_id,
        "history_event_ids_json": json.dumps([event.event_id for event in history]),
        "history_item_ids_json": json.dumps([event.item_id for event in history]),
        "history_length": len(history),
        "history_timestamps_json": json.dumps([event.timestamp for event in history]),
        "target_event_id": target.event_id,
        "target_item_id": target.item_id,
        "target_split": target_split,
        "target_timestamp": target.timestamp,
        "user_id": user_id,
    }


def _sequence_row(
    *,
    user_id: int,
    train_events: list[_HistoryEvent],
    val_events: list[_HistoryEvent],
    test_events: list[_HistoryEvent],
) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "train_event_ids_json": _event_field(train_events, "event_id"),
        "train_item_ids_json": _event_field(train_events, "item_id"),
        "train_length": len(train_events),
        "train_timestamps_json": _event_field(train_events, "timestamp"),
        "validation_event_ids_json": _event_field(val_events, "event_id"),
        "validation_item_ids_json": _event_field(val_events, "item_id"),
        "validation_length": len(val_events),
        "validation_timestamps_json": _event_field(val_events, "timestamp"),
        "test_event_ids_json": _event_field(test_events, "event_id"),
        "test_item_ids_json": _event_field(test_events, "item_id"),
        "test_length": len(test_events),
        "test_timestamps_json": _event_field(test_events, "timestamp"),
    }


def _event_field(events: list[_HistoryEvent], field: str) -> str:
    return json.dumps([getattr(event, field) for event in events])


def _write_rows(path: Path, rows: list[dict[str, Any]], *, id_field: str) -> None:
    fieldnames = [
        id_field,
        "user_id",
        "target_event_id",
        "target_item_id",
        "target_timestamp",
        "target_split",
        "history_length",
        "history_item_ids_json",
        "history_timestamps_json",
        "history_event_ids_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_user_sequences(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "user_id",
        "train_length",
        "train_item_ids_json",
        "train_timestamps_json",
        "train_event_ids_json",
        "validation_length",
        "validation_item_ids_json",
        "validation_timestamps_json",
        "validation_event_ids_json",
        "test_length",
        "test_item_ids_json",
        "test_timestamps_json",
        "test_event_ids_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _default_dataset_name(dataset_root: Path, split_name: str) -> str:
    stem = dataset_root.name.lower().replace("-", "_")
    suffix = "loo" if split_name == "temporal_leave_one_out" else "global_time"
    return f"{stem}_{suffix}_sequence_cases"


def _write_readme(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Sequence Case Export",
                "",
                "This artifact contains compact user sequences and leakage-safe validation/test",
                "cases. `train_examples.csv` is header-only unless the export was run with",
                "`--write-train-examples`; runners can materialize train prefixes from",
                "`user_sequences.csv` as needed. Do not add validation/test target events to",
                "their own histories.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
