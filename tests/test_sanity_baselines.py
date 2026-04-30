import json
from pathlib import Path

import pandas as pd
import pytest

from tglrec.cli import main
from tglrec.data import schema
from tglrec.eval.metrics import rank_by_score
from tglrec.models.sanity_baselines import IncrementalTrainingStats, run_sanity_baselines


def _write_processed_dataset(root: Path, interactions: list[dict[str, object]], item_ids: list[int]) -> None:
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
    pd.DataFrame(
        {
            schema.ITEM_ID: item_ids,
            schema.RAW_ITEM_ID: [str(item_id) for item_id in item_ids],
            "title": [f"Item {item_id}" for item_id in item_ids],
            "genres": ["Synthetic"] * len(item_ids),
        }
    ).to_csv(root / "items.csv", index=False)


def test_incremental_item_knn_scores_tied_neighbors_deterministically():
    stats = IncrementalTrainingStats()
    stats.add_event(user_id=1, item_id=3, timestamp=100)
    stats.add_event(user_id=1, item_id=2, timestamp=101)
    stats.add_event(user_id=2, item_id=3, timestamp=102)
    stats.add_event(user_id=2, item_id=10, timestamp=103)
    stats.add_user_history_event(user_id=0, item_id=3, timestamp=104)

    scores = stats.item_knn_scores(user_id=0, candidates={2, 10}, neighbors=50)

    assert scores == {2: 1.0, 10: 1.0}
    assert rank_by_score(scores) == [2, 10]


def test_sanity_baselines_use_only_train_events_before_target_timestamp(tmp_path: Path):
    dataset = tmp_path / "processed"
    _write_processed_dataset(
        dataset,
        [
            {
                schema.EVENT_ID: 0,
                schema.USER_ID: 0,
                schema.ITEM_ID: 0,
                schema.TIMESTAMP: 100,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 1,
                schema.USER_ID: 0,
                schema.ITEM_ID: 9,
                schema.TIMESTAMP: 200,
                schema.SPLIT_LOO: "test",
            },
            {
                schema.EVENT_ID: 2,
                schema.USER_ID: 1,
                schema.ITEM_ID: 9,
                schema.TIMESTAMP: 300,
                schema.SPLIT_LOO: "train",
            },
        ],
        item_ids=[0, 1, 9],
    )

    result = run_sanity_baselines(
        dataset_dir=dataset,
        output_dir=tmp_path / "run",
        ks=(1, 3),
        command="synthetic",
    )

    assert result.metrics["popularity"]["HR@1"] == pytest.approx(0.0)
    assert result.metrics["popularity"]["HR@3"] == pytest.approx(1.0)


def test_sanity_baselines_use_validation_as_test_history_only(tmp_path: Path):
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
                schema.USER_ID: 1,
                schema.ITEM_ID: 2,
                schema.TIMESTAMP: 110,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 2,
                schema.USER_ID: 2,
                schema.ITEM_ID: 3,
                schema.TIMESTAMP: 120,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 3,
                schema.USER_ID: 0,
                schema.ITEM_ID: 2,
                schema.TIMESTAMP: 150,
                schema.SPLIT_LOO: "val",
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
    )

    result = run_sanity_baselines(
        dataset_dir=dataset,
        output_dir=tmp_path / "run",
        ks=(1,),
        command="synthetic",
    )
    train_only = run_sanity_baselines(
        dataset_dir=dataset,
        output_dir=tmp_path / "run_train_only",
        ks=(1,),
        use_validation_history_for_test=False,
        command="synthetic",
    )

    assert result.metrics["popularity"]["HR@1"] == pytest.approx(1.0)
    assert train_only.metrics["popularity"]["HR@1"] == pytest.approx(0.0)
    config = (result.output_dir / "config.yaml").read_text(encoding="utf-8")
    assert "val_user_history_only" in config


def test_sanity_baselines_evaluate_multiple_global_time_cases_per_user(tmp_path: Path):
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
                schema.SPLIT_GLOBAL: "train",
            },
            {
                schema.EVENT_ID: 1,
                schema.USER_ID: 0,
                schema.ITEM_ID: 2,
                schema.TIMESTAMP: 200,
                schema.SPLIT_LOO: "test",
                schema.SPLIT_GLOBAL: "test",
            },
            {
                schema.EVENT_ID: 2,
                schema.USER_ID: 0,
                schema.ITEM_ID: 3,
                schema.TIMESTAMP: 300,
                schema.SPLIT_LOO: "test",
                schema.SPLIT_GLOBAL: "test",
            },
        ],
        item_ids=[1, 2, 3],
    )

    result = run_sanity_baselines(
        dataset_dir=dataset,
        output_dir=tmp_path / "run",
        split_name="global_time",
        ks=(3,),
        command="synthetic",
    )

    metrics = json.loads((result.output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert result.num_cases == 2
    assert metrics["num_eval_cases"] == 2
    assert metrics["num_eval_users"] == 1


def test_sanity_baseline_cli_smoke_writes_required_run_files(tmp_path: Path, capsys):
    dataset = tmp_path / "processed"
    output = tmp_path / "run"
    _write_processed_dataset(
        dataset,
        [
            {
                schema.EVENT_ID: 0,
                schema.USER_ID: 0,
                schema.ITEM_ID: 2,
                schema.TIMESTAMP: 100,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 1,
                schema.USER_ID: 1,
                schema.ITEM_ID: 10,
                schema.TIMESTAMP: 100,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 2,
                schema.USER_ID: 0,
                schema.ITEM_ID: 10,
                schema.TIMESTAMP: 200,
                schema.SPLIT_LOO: "test",
            },
        ],
        item_ids=[2, 10],
    )

    assert (
        main(
            [
                "evaluate",
                "sanity-baselines",
                "--dataset-dir",
                str(dataset),
                "--output-dir",
                str(output),
                "--ks",
                "1",
                "2",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()

    assert "popularity:" in captured.out
    for name in [
        "config.yaml",
        "metrics.json",
        "metrics_by_segment.csv",
        "command.txt",
        "git_commit.txt",
        "stdout.log",
        "stderr.log",
        "environment.json",
    ]:
        assert (output / name).exists()
