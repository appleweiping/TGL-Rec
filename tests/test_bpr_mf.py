import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from tglrec.cli import main
from tglrec.data import schema
from tglrec.models.bpr_mf import run_bpr_mf


def _write_processed_dataset(root: Path, interactions: list[dict[str, object]]) -> None:
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
    item_ids = sorted(int(item_id) for item_id in frame[schema.ITEM_ID].unique())
    pd.DataFrame(
        [
            {
                schema.ITEM_ID: item_id,
                schema.RAW_ITEM_ID: str(item_id),
                "title": f"Item {item_id}",
            }
            for item_id in item_ids
        ]
    ).to_csv(root / "items.csv", index=False)


def test_bpr_mf_cli_writes_deterministic_run_outputs(tmp_path: Path, capsys):
    dataset = tmp_path / "processed"
    _write_processed_dataset(
        dataset,
        [
            {schema.EVENT_ID: 0, schema.USER_ID: 1, schema.ITEM_ID: 1, schema.TIMESTAMP: 10, schema.SPLIT_LOO: "train"},
            {schema.EVENT_ID: 1, schema.USER_ID: 1, schema.ITEM_ID: 2, schema.TIMESTAMP: 20, schema.SPLIT_LOO: "train"},
            {schema.EVENT_ID: 2, schema.USER_ID: 2, schema.ITEM_ID: 1, schema.TIMESTAMP: 10, schema.SPLIT_LOO: "train"},
            {schema.EVENT_ID: 3, schema.USER_ID: 2, schema.ITEM_ID: 3, schema.TIMESTAMP: 20, schema.SPLIT_LOO: "train"},
            {schema.EVENT_ID: 4, schema.USER_ID: 0, schema.ITEM_ID: 1, schema.TIMESTAMP: 30, schema.SPLIT_LOO: "train"},
            {schema.EVENT_ID: 5, schema.USER_ID: 0, schema.ITEM_ID: 2, schema.TIMESTAMP: 40, schema.SPLIT_LOO: "val"},
            {schema.EVENT_ID: 6, schema.USER_ID: 0, schema.ITEM_ID: 3, schema.TIMESTAMP: 50, schema.SPLIT_LOO: "test"},
            {schema.EVENT_ID: 7, schema.USER_ID: 3, schema.ITEM_ID: 4, schema.TIMESTAMP: 10, schema.SPLIT_LOO: "train"},
            {schema.EVENT_ID: 8, schema.USER_ID: 3, schema.ITEM_ID: 3, schema.TIMESTAMP: 20, schema.SPLIT_LOO: "train"},
        ],
    )
    first = tmp_path / "first"
    second = tmp_path / "second"

    assert (
        main(
            [
                "train",
                "bpr-mf",
                "--dataset-dir",
                str(dataset),
                "--output-dir",
                str(first),
                "--ks",
                "1",
                "5",
                "--factors",
                "4",
                "--epochs",
                "2",
                "--learning-rate",
                "0.03",
                "--max-train-pairs",
                "8",
                "--seed",
                "2026",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "wrote BPR-MF run:" in captured.out
    run_bpr_mf(
        dataset_dir=dataset,
        output_dir=second,
        ks=(1, 5),
        factors=4,
        epochs=2,
        learning_rate=0.03,
        max_train_pairs=8,
        seed=2026,
        command="synthetic",
    )

    for name in [
        "config.yaml",
        "metrics.json",
        "metrics_by_epoch.csv",
        "metrics_by_case.csv",
        "metrics_by_segment.csv",
        "command.txt",
        "git_commit.txt",
        "git_status.txt",
        "run_status.json",
        "stdout.log",
        "stderr.log",
        "environment.json",
        "checksums.json",
    ]:
        assert (first / name).exists()
    assert (first / "metrics_by_case.csv").read_text(encoding="utf-8") == (
        second / "metrics_by_case.csv"
    ).read_text(encoding="utf-8")
    assert (first / "metrics_by_epoch.csv").read_text(encoding="utf-8") == (
        second / "metrics_by_epoch.csv"
    ).read_text(encoding="utf-8")
    config = (first / "config.yaml").read_text(encoding="utf-8")
    assert "baseline_name: bpr_mf" in config
    assert "max_train_pairs: 8" in config
    metrics = json.loads((first / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["baseline"] == "bpr_mf"
    assert metrics["candidate_mode"] == "full_ranking"
    assert set(metrics["metrics"]) >= {"HR@1", "HR@5", "NDCG@5", "MRR@5"}
    with (first / "metrics_by_case.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["candidate_count"] == "2"
    assert 2 not in json.loads(rows[0]["top_item_ids_json"])


def test_bpr_mf_rejects_bad_inputs_and_missing_item_metadata(tmp_path: Path):
    dataset = tmp_path / "processed"
    _write_processed_dataset(
        dataset,
        [
            {schema.EVENT_ID: 0, schema.USER_ID: 0, schema.ITEM_ID: 1, schema.TIMESTAMP: 10, schema.SPLIT_LOO: "train"},
            {schema.EVENT_ID: 1, schema.USER_ID: 0, schema.ITEM_ID: 2, schema.TIMESTAMP: 20, schema.SPLIT_LOO: "test"},
        ],
    )

    with pytest.raises(ValueError, match="factors must be positive"):
        run_bpr_mf(dataset_dir=dataset, output_dir=tmp_path / "bad", factors=0)
    with pytest.raises(ValueError, match="positive cutoffs"):
        run_bpr_mf(dataset_dir=dataset, output_dir=tmp_path / "bad_k", ks=(0,))

    items = pd.read_csv(dataset / "items.csv")
    items = items.loc[items[schema.ITEM_ID] != 2]
    items.to_csv(dataset / "items.csv", index=False)
    with pytest.raises(ValueError, match="items.csv is missing"):
        run_bpr_mf(dataset_dir=dataset, output_dir=tmp_path / "missing_item")
