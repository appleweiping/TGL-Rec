import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from tglrec.cli import main
from tglrec.data import schema
from tglrec.eval.tdig_recall import run_tdig_candidate_recall


def _write_processed_dataset(
    root: Path,
    interactions: list[dict[str, object]],
    item_ids: list[int],
    item_metadata: dict[int, dict[str, object]] | None = None,
) -> None:
    root.mkdir(parents=True)
    frame = pd.DataFrame(interactions)
    for column in schema.INTERACTION_COLUMNS:
        if column not in frame.columns:
            if column == schema.SPLIT_GLOBAL:
                frame[column] = frame[schema.SPLIT_LOO]
            elif column == schema.RATING:
                frame[column] = 1.0
            elif column == schema.RAW_USER_ID:
                frame[column] = frame[schema.USER_ID].astype(str)
            elif column == schema.RAW_ITEM_ID:
                frame[column] = frame[schema.ITEM_ID].astype(str)
            else:
                raise AssertionError(f"missing required synthetic column {column}")
    frame[schema.INTERACTION_COLUMNS].to_csv(root / "interactions.csv", index=False)
    items: list[dict[str, object]] = []
    for item_id in item_ids:
        row = {
            schema.ITEM_ID: item_id,
            schema.RAW_ITEM_ID: str(item_id),
            "title": f"Item {item_id}",
            "genres": "Synthetic",
        }
        if item_metadata is not None and item_id in item_metadata:
            row.update(item_metadata[item_id])
        items.append(row)
    pd.DataFrame(items).to_csv(root / "items.csv", index=False)


def test_tdig_candidate_recall_uses_only_train_edges_strictly_before_target(tmp_path: Path):
    dataset = tmp_path / "processed"
    _write_processed_dataset(
        dataset,
        [
            {
                schema.EVENT_ID: 0,
                schema.USER_ID: 2,
                schema.ITEM_ID: 1,
                schema.TIMESTAMP: 50,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 1,
                schema.USER_ID: 2,
                schema.ITEM_ID: 3,
                schema.TIMESTAMP: 60,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 2,
                schema.USER_ID: 0,
                schema.ITEM_ID: 1,
                schema.TIMESTAMP: 100,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 3,
                schema.USER_ID: 0,
                schema.ITEM_ID: 2,
                schema.TIMESTAMP: 200,
                schema.SPLIT_LOO: "test",
            },
            {
                schema.EVENT_ID: 4,
                schema.USER_ID: 1,
                schema.ITEM_ID: 1,
                schema.TIMESTAMP: 300,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 5,
                schema.USER_ID: 1,
                schema.ITEM_ID: 2,
                schema.TIMESTAMP: 400,
                schema.SPLIT_LOO: "train",
            },
        ],
        item_ids=[1, 2, 3],
        item_metadata={
            1: {"title": "Camera starter", "genres": "gear"},
            2: {"title": "Camera upgrade", "genres": "gear"},
            3: {"title": "Piano course", "genres": "music"},
        },
    )

    result = run_tdig_candidate_recall(
        dataset_dir=dataset,
        output_dir=tmp_path / "run",
        ks=(1, 2),
        source_history_items=1,
        command="synthetic",
    )
    with (result.output_dir / "metrics_by_case.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert result.metrics["candidate_recall@1"] == pytest.approx(0.0)
    assert result.metrics["candidate_recall@2"] == pytest.approx(0.0)
    assert json.loads(rows[0]["top_candidate_ids_json"]) == [3]
    assert rows[0]["target_rank"] == ""
    assert rows[0]["semantic_vs_transition_case_type"] == "semantic_only"
    assert rows[0]["target_has_transition_evidence"] == "0"
    assert json.loads(rows[0]["semantic_overlap_tokens_json"]) == ["camera", "gear"]


def test_tdig_candidate_recall_can_use_validation_event_as_test_source(tmp_path: Path):
    dataset = tmp_path / "processed"
    _write_processed_dataset(
        dataset,
        [
            {
                schema.EVENT_ID: 0,
                schema.USER_ID: 1,
                schema.ITEM_ID: 2,
                schema.TIMESTAMP: 50,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 1,
                schema.USER_ID: 1,
                schema.ITEM_ID: 3,
                schema.TIMESTAMP: 60,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 2,
                schema.USER_ID: 0,
                schema.ITEM_ID: 1,
                schema.TIMESTAMP: 100,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 3,
                schema.USER_ID: 0,
                schema.ITEM_ID: 2,
                schema.TIMESTAMP: 200,
                schema.SPLIT_LOO: "val",
            },
            {
                schema.EVENT_ID: 4,
                schema.USER_ID: 0,
                schema.ITEM_ID: 3,
                schema.TIMESTAMP: 300,
                schema.SPLIT_LOO: "test",
            },
        ],
        item_ids=[1, 2, 3],
        item_metadata={
            1: {"title": "Tripod", "genres": "photo"},
            2: {"title": "Coffee beans", "genres": "grocery"},
            3: {"title": "Running socks", "genres": "fitness"},
        },
    )

    with_validation = run_tdig_candidate_recall(
        dataset_dir=dataset,
        output_dir=tmp_path / "with_validation",
        ks=(1,),
        source_history_items=1,
        command="synthetic",
    )
    train_only = run_tdig_candidate_recall(
        dataset_dir=dataset,
        output_dir=tmp_path / "train_only",
        ks=(1,),
        source_history_items=1,
        use_validation_history_for_test=False,
        command="synthetic",
    )

    assert with_validation.metrics["candidate_recall@1"] == pytest.approx(1.0)
    assert train_only.metrics["candidate_recall@1"] == pytest.approx(0.0)
    assert "val_user_history_only" in (with_validation.output_dir / "config.yaml").read_text()
    with (with_validation.output_dir / "metrics_by_case.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["semantic_vs_transition_case_type"] == "transition_only"
    assert rows[0]["target_has_transition_evidence"] == "1"


def test_tdig_candidate_recall_skips_ambiguous_same_timestamp_source_chain(tmp_path: Path):
    dataset = tmp_path / "processed"
    _write_processed_dataset(
        dataset,
        [
            {
                schema.EVENT_ID: 0,
                schema.USER_ID: 1,
                schema.ITEM_ID: 1,
                schema.TIMESTAMP: 50,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 1,
                schema.USER_ID: 1,
                schema.ITEM_ID: 2,
                schema.TIMESTAMP: 50,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 2,
                schema.USER_ID: 1,
                schema.ITEM_ID: 3,
                schema.TIMESTAMP: 60,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 3,
                schema.USER_ID: 0,
                schema.ITEM_ID: 2,
                schema.TIMESTAMP: 100,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 4,
                schema.USER_ID: 0,
                schema.ITEM_ID: 3,
                schema.TIMESTAMP: 200,
                schema.SPLIT_LOO: "test",
            },
        ],
        item_ids=[1, 2, 3],
        item_metadata={
            1: {"title": "Space opera starter", "genres": "cosmos"},
            2: {"title": "Space opera sequel", "genres": "cosmos"},
            3: {"title": "Cooking memoir", "genres": "food"},
        },
    )

    result = run_tdig_candidate_recall(
        dataset_dir=dataset,
        output_dir=tmp_path / "run",
        ks=(10,),
        source_history_items=1,
        command="synthetic",
    )

    assert result.metrics["candidate_recall@10"] == pytest.approx(0.0)
    assert result.metrics["same_timestamp_tie_group_skip_count"] == pytest.approx(1.0)
    assert result.metrics["same_timestamp_adjacent_transition_skip_count"] == pytest.approx(1.0)
    assert result.metrics["same_timestamp_ambiguous_bridge_skip_count"] == pytest.approx(1.0)


def test_tdig_candidate_recall_does_not_use_same_timestamp_validation_source(tmp_path: Path):
    dataset = tmp_path / "processed"
    _write_processed_dataset(
        dataset,
        [
            {
                schema.EVENT_ID: 0,
                schema.USER_ID: 1,
                schema.ITEM_ID: 2,
                schema.TIMESTAMP: 50,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 1,
                schema.USER_ID: 1,
                schema.ITEM_ID: 3,
                schema.TIMESTAMP: 60,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 2,
                schema.USER_ID: 0,
                schema.ITEM_ID: 1,
                schema.TIMESTAMP: 100,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 3,
                schema.USER_ID: 0,
                schema.ITEM_ID: 2,
                schema.TIMESTAMP: 300,
                schema.SPLIT_LOO: "val",
            },
            {
                schema.EVENT_ID: 4,
                schema.USER_ID: 0,
                schema.ITEM_ID: 3,
                schema.TIMESTAMP: 300,
                schema.SPLIT_LOO: "test",
            },
        ],
        item_ids=[1, 2, 3],
        item_metadata={
            1: {"title": "Space opera starter", "genres": "cosmos"},
            2: {"title": "Space opera sequel", "genres": "cosmos"},
            3: {"title": "Cooking memoir", "genres": "food"},
        },
    )

    result = run_tdig_candidate_recall(
        dataset_dir=dataset,
        output_dir=tmp_path / "run",
        ks=(1,),
        source_history_items=1,
        command="synthetic",
    )

    assert result.metrics["candidate_recall@1"] == pytest.approx(0.0)


def test_tdig_candidate_recall_labels_transition_when_target_metadata_is_empty(tmp_path: Path):
    dataset = tmp_path / "processed"
    _write_processed_dataset(
        dataset,
        [
            {
                schema.EVENT_ID: 0,
                schema.USER_ID: 1,
                schema.ITEM_ID: 1,
                schema.TIMESTAMP: 10,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 1,
                schema.USER_ID: 1,
                schema.ITEM_ID: 2,
                schema.TIMESTAMP: 20,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 2,
                schema.USER_ID: 0,
                schema.ITEM_ID: 1,
                schema.TIMESTAMP: 30,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 3,
                schema.USER_ID: 0,
                schema.ITEM_ID: 2,
                schema.TIMESTAMP: 40,
                schema.SPLIT_LOO: "test",
            },
        ],
        item_ids=[1, 2],
        item_metadata={
            1: {"title": "Tripod mount", "genres": "camera"},
            2: {"title": None, "genres": None},
        },
    )

    result = run_tdig_candidate_recall(
        dataset_dir=dataset,
        output_dir=tmp_path / "run",
        ks=(1,),
        source_history_items=1,
        command="synthetic",
    )

    with (result.output_dir / "metrics_by_case.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert result.metrics["candidate_recall@1"] == pytest.approx(1.0)
    assert rows[0]["semantic_vs_transition_case_type"] == "transition_only"
    assert rows[0]["semantic_overlap_max"] == "0"
    assert rows[0]["semantic_overlap_tokens_json"] == "[]"


def test_tdig_candidate_recall_rejects_invalid_cutoffs_and_item_universe(tmp_path: Path):
    dataset = tmp_path / "processed"
    _write_processed_dataset(
        dataset,
        [
            {
                schema.EVENT_ID: 0,
                schema.USER_ID: 0,
                schema.ITEM_ID: 1,
                schema.TIMESTAMP: 100,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 1,
                schema.USER_ID: 0,
                schema.ITEM_ID: 2,
                schema.TIMESTAMP: 200,
                schema.SPLIT_LOO: "test",
            },
        ],
        item_ids=[1, 2],
    )

    with pytest.raises(ValueError, match="positive cutoffs"):
        run_tdig_candidate_recall(dataset_dir=dataset, output_dir=tmp_path / "bad_k", ks=(0,))

    _write_processed_dataset(
        tmp_path / "missing_item",
        [
            {
                schema.EVENT_ID: 0,
                schema.USER_ID: 0,
                schema.ITEM_ID: 1,
                schema.TIMESTAMP: 100,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 1,
                schema.USER_ID: 0,
                schema.ITEM_ID: 2,
                schema.TIMESTAMP: 200,
                schema.SPLIT_LOO: "test",
            },
        ],
        item_ids=[1],
    )
    with pytest.raises(ValueError, match="items.csv is missing"):
        run_tdig_candidate_recall(
            dataset_dir=tmp_path / "missing_item",
            output_dir=tmp_path / "bad_items",
            ks=(1,),
        )


def test_tdig_candidate_recall_is_deterministic_and_writes_schema(tmp_path: Path, capsys):
    dataset = tmp_path / "processed"
    _write_processed_dataset(
        dataset,
        [
            {
                schema.EVENT_ID: 0,
                schema.USER_ID: 1,
                schema.ITEM_ID: 1,
                schema.TIMESTAMP: 10,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 1,
                schema.USER_ID: 1,
                schema.ITEM_ID: 2,
                schema.TIMESTAMP: 20,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 2,
                schema.USER_ID: 2,
                schema.ITEM_ID: 1,
                schema.TIMESTAMP: 10,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 3,
                schema.USER_ID: 2,
                schema.ITEM_ID: 3,
                schema.TIMESTAMP: 20,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 4,
                schema.USER_ID: 0,
                schema.ITEM_ID: 1,
                schema.TIMESTAMP: 30,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 5,
                schema.USER_ID: 0,
                schema.ITEM_ID: 2,
                schema.TIMESTAMP: 40,
                schema.SPLIT_LOO: "test",
            },
        ],
        item_ids=[1, 2, 3],
        item_metadata={
            1: {"title": "Space opera starter", "genres": "cosmos"},
            2: {"title": "Space opera sequel", "genres": "cosmos"},
            3: {"title": "Cooking memoir", "genres": "food"},
        },
    )
    first = tmp_path / "first"
    second = tmp_path / "second"

    assert (
        main(
            [
                "evaluate",
                "tdig-candidate-recall",
                "--dataset-dir",
                str(dataset),
                "--output-dir",
                str(first),
                "--ks",
                "1",
                "2",
                "--max-history-items",
                "1",
                "--per-source-top-k",
                "1",
                "--aggregation",
                "sum",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "wrote TDIG candidate recall run:" in captured.out

    run_tdig_candidate_recall(
        dataset_dir=dataset,
        output_dir=second,
        ks=(1, 2),
        max_history_items=1,
        per_source_top_k=1,
        aggregation="sum",
        command="synthetic",
    )

    for name in [
        "config.yaml",
        "metrics.json",
        "metrics_by_case.csv",
        "metrics_by_segment.csv",
        "command.txt",
        "git_commit.txt",
        "stdout.log",
        "stderr.log",
        "environment.json",
        "git_status.txt",
        "run_status.json",
        "checksums.json",
    ]:
        assert (first / name).exists()
    metrics = json.loads((first / "metrics.json").read_text(encoding="utf-8"))
    config = (first / "config.yaml").read_text(encoding="utf-8")
    assert metrics["evaluator"] == "tdig_direct_candidate_recall"
    assert metrics["candidate_mode"] == "tdig_direct_transition_recall"
    assert metrics["metrics"]["candidate_recall@1"] == pytest.approx(1.0)
    assert "per_source_top_k: 1" in config
    assert "aggregation: sum" in config
    assert "dataset_provenance:" in config
    assert "same_timestamp_skip_metric_definitions:" in config
    assert "semantic_vs_transition_labeling:" in config
    assert "strictly before the test target" in config
    assert "skipped_same_timestamp_transitions" not in json.dumps(metrics)
    assert "same_timestamp_tie_group_skip_count" in json.dumps(metrics)
    assert "same_timestamp_adjacent_transition_skip_count" in json.dumps(metrics)
    checksums = json.loads((first / "checksums.json").read_text(encoding="utf-8"))
    assert "metrics_by_case.csv" in checksums["files"]
    assert "checksums.json" not in checksums["files"]
    assert (first / "metrics_by_case.csv").read_text(encoding="utf-8") == (
        second / "metrics_by_case.csv"
    ).read_text(encoding="utf-8")
    with (first / "metrics_by_case.csv").open(newline="") as handle:
        case_rows = list(csv.DictReader(handle))
    assert case_rows[0]["semantic_vs_transition_case_type"] == "semantic_and_transition"
    assert case_rows[0]["target_has_transition_evidence"] == "1"
    assert json.loads(case_rows[0]["semantic_overlap_tokens_json"]) == [
        "cosmos",
        "opera",
        "space",
    ]
    with (first / "metrics_by_segment.csv").open(newline="") as handle:
        segment_rows = list(csv.DictReader(handle))
    assert {
        (row["segment_name"], row["segment_value"]) for row in segment_rows
    } >= {("semantic_vs_transition_case_type", "semantic_and_transition")}
