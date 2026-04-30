import json
from pathlib import Path

import pandas as pd
import pytest

from tglrec.cli import main
from tglrec.data import schema
from tglrec.eval.history_perturbations import (
    HistoryEvent,
    _perturb_history_events,
    run_history_perturbation_diagnostics,
)
from tglrec.models.sanity_baselines import EvaluationCase


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


def _diagnostic_dataset(root: Path) -> Path:
    dataset = root / "processed"
    _write_processed_dataset(
        dataset,
        [
            {
                schema.EVENT_ID: 0,
                schema.USER_ID: 10,
                schema.ITEM_ID: 2,
                schema.TIMESTAMP: 10,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 1,
                schema.USER_ID: 10,
                schema.ITEM_ID: 3,
                schema.TIMESTAMP: 11,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 2,
                schema.USER_ID: 11,
                schema.ITEM_ID: 1,
                schema.TIMESTAMP: 10,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 3,
                schema.USER_ID: 11,
                schema.ITEM_ID: 4,
                schema.TIMESTAMP: 11,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 4,
                schema.USER_ID: 0,
                schema.ITEM_ID: 1,
                schema.TIMESTAMP: 100,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 5,
                schema.USER_ID: 0,
                schema.ITEM_ID: 2,
                schema.TIMESTAMP: 200,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 6,
                schema.USER_ID: 0,
                schema.ITEM_ID: 3,
                schema.TIMESTAMP: 300,
                schema.SPLIT_LOO: "test",
            },
        ],
        item_ids=[1, 2, 3, 4],
    )
    return dataset


def _timestamp_diagnostic_dataset(root: Path) -> Path:
    day = 24 * 60 * 60
    dataset = root / "timestamp_processed"
    _write_processed_dataset(
        dataset,
        [
            {
                schema.EVENT_ID: 0,
                schema.USER_ID: 0,
                schema.ITEM_ID: 1,
                schema.TIMESTAMP: 10 * day,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 1,
                schema.USER_ID: 0,
                schema.ITEM_ID: 2,
                schema.TIMESTAMP: 20 * day,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 2,
                schema.USER_ID: 0,
                schema.ITEM_ID: 3,
                schema.TIMESTAMP: 95 * day,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 3,
                schema.USER_ID: 0,
                schema.ITEM_ID: 4,
                schema.TIMESTAMP: 98 * day,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 4,
                schema.USER_ID: 0,
                schema.ITEM_ID: 99,
                schema.TIMESTAMP: 100 * day,
                schema.SPLIT_LOO: "test",
            },
        ],
        item_ids=[1, 2, 3, 4, 99],
    )
    return dataset


def test_history_perturbation_diagnostics_preserve_scoring_window_items(tmp_path: Path):
    dataset = _diagnostic_dataset(tmp_path)

    result = run_history_perturbation_diagnostics(
        dataset_dir=dataset,
        output_dir=tmp_path / "run",
        ks=(1,),
        item_knn_max_history_items=1,
        seed=2026,
        command="synthetic",
    )

    assert result.metrics["item_knn"]["original"]["HR@1"] == pytest.approx(1.0)
    assert result.metrics["item_knn"]["history_shuffle"]["HR@1"] == pytest.approx(1.0)
    assert result.metrics["item_knn"]["order_reversal"]["HR@1"] == pytest.approx(1.0)
    assert result.metrics["item_knn"]["timestamp_removal"]["HR@1"] == pytest.approx(1.0)
    assert result.metrics["item_knn"]["timestamp_randomization"]["HR@1"] == pytest.approx(1.0)
    assert result.metrics["item_knn"]["window_swap"]["HR@1"] == pytest.approx(1.0)
    delta = result.deltas["item_knn"]["order_reversal"]["HR@1"]
    assert delta["delta_from_original"] == pytest.approx(0.0)
    assert delta["sensitivity_index"] == pytest.approx(0.0)

    metrics_json = json.loads((result.output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_json["diagnostic_name"] == "history_perturbations"
    assert metrics_json["baselines"]["popularity"]["original"] == metrics_json["baselines"]["popularity"][
        "order_reversal"
    ]
    assert metrics_json["baselines"]["popularity"]["original"] == metrics_json["baselines"]["popularity"][
        "timestamp_removal"
    ]
    case_metrics = pd.read_csv(result.output_dir / "metrics_by_case.csv")
    assert set(case_metrics["model"]) == {"popularity", "item_knn"}
    assert set(case_metrics["perturbation"]) == {
        "history_shuffle",
        "order_reversal",
        "timestamp_removal",
        "timestamp_randomization",
        "window_swap",
    }
    assert {
        "user_id",
        "target_item_id",
        "model",
        "perturbation",
        "original_rank",
        "perturbed_rank",
        "rank_delta",
        "hit_delta@1",
    }.issubset(case_metrics.columns)
    item_knn_rows = case_metrics.loc[case_metrics["model"] == "item_knn"]
    assert set(item_knn_rows["original_rank"]) == {1}
    assert set(item_knn_rows["perturbed_rank"]) == {1}
    assert set(item_knn_rows["rank_delta"]) == {0}
    assert set(item_knn_rows["original_hit@1"]) == {1}
    assert set(item_knn_rows["perturbed_hit@1"]) == {1}
    assert set(item_knn_rows["hit_delta@1"]) == {0}


def test_history_perturbation_outputs_timestamp_audit_fields(tmp_path: Path):
    dataset = _timestamp_diagnostic_dataset(tmp_path)

    result = run_history_perturbation_diagnostics(
        dataset_dir=dataset,
        output_dir=tmp_path / "run",
        ks=(1,),
        item_knn_max_history_items=0,
        seed=2026,
        command="synthetic",
    )

    case_metrics = pd.read_csv(result.output_dir / "metrics_by_case.csv")
    assert {
        "history_event_count",
        "item_position_changed_count",
        "timestamp_changed_count",
        "original_timestamp_null_count",
        "perturbed_timestamp_null_count",
        "original_history_fingerprint",
        "perturbed_history_fingerprint",
    }.issubset(case_metrics.columns)

    item_knn_rows = case_metrics.loc[case_metrics["model"] == "item_knn"].set_index("perturbation")
    assert int(item_knn_rows.loc["timestamp_removal", "history_event_count"]) == 4
    assert int(item_knn_rows.loc["timestamp_removal", "item_position_changed_count"]) == 0
    assert int(item_knn_rows.loc["timestamp_removal", "timestamp_changed_count"]) == 4
    assert int(item_knn_rows.loc["timestamp_removal", "perturbed_timestamp_null_count"]) == 4
    assert int(item_knn_rows.loc["timestamp_randomization", "item_position_changed_count"]) == 0
    assert int(item_knn_rows.loc["timestamp_randomization", "timestamp_changed_count"]) > 0
    assert int(item_knn_rows.loc["window_swap", "item_position_changed_count"]) == 0
    assert int(item_knn_rows.loc["window_swap", "timestamp_changed_count"]) == 4
    assert (
        item_knn_rows.loc["window_swap", "original_history_fingerprint"]
        != item_knn_rows.loc["window_swap", "perturbed_history_fingerprint"]
    )


def test_timestamp_removal_preserves_items_and_removes_time_signal():
    case = EvaluationCase(user_id=7, item_id=99, timestamp=40, event_id=10)
    events = [
        HistoryEvent(user_id=7, item_id=1, timestamp=10, event_id=1),
        HistoryEvent(user_id=7, item_id=2, timestamp=20, event_id=2),
        HistoryEvent(user_id=7, item_id=3, timestamp=30, event_id=3),
    ]

    perturbed = _perturb_history_events(events, "timestamp_removal", case=case, seed=2026)

    assert [event.item_id for event in perturbed] == [1, 2, 3]
    assert [event.event_id for event in perturbed] == [1, 2, 3]
    assert [event.timestamp for event in perturbed] == [None, None, None]


def test_timestamp_randomization_is_deterministic_and_uses_only_history_timestamps():
    case = EvaluationCase(user_id=7, item_id=99, timestamp=40, event_id=10)
    events = [
        HistoryEvent(user_id=7, item_id=1, timestamp=10, event_id=1),
        HistoryEvent(user_id=7, item_id=2, timestamp=20, event_id=2),
        HistoryEvent(user_id=7, item_id=3, timestamp=30, event_id=3),
        HistoryEvent(user_id=7, item_id=4, timestamp=35, event_id=4),
    ]

    first = _perturb_history_events(events, "timestamp_randomization", case=case, seed=2026)
    second = _perturb_history_events(events, "timestamp_randomization", case=case, seed=2026)

    assert first == second
    assert [event.item_id for event in first] == [1, 2, 3, 4]
    assert sorted(event.timestamp for event in first) == [10, 20, 30, 35]
    assert [event.timestamp for event in first] != [10, 20, 30, 35]
    assert all(event.timestamp is not None and event.timestamp < case.timestamp for event in first)


def test_window_swap_exchanges_within_week_and_long_gap_timestamps_without_reordering_items():
    day = 24 * 60 * 60
    case = EvaluationCase(user_id=7, item_id=99, timestamp=100 * day, event_id=10)
    events = [
        HistoryEvent(user_id=7, item_id=1, timestamp=10 * day, event_id=1),
        HistoryEvent(user_id=7, item_id=2, timestamp=20 * day, event_id=2),
        HistoryEvent(user_id=7, item_id=3, timestamp=95 * day, event_id=3),
        HistoryEvent(user_id=7, item_id=4, timestamp=98 * day, event_id=4),
    ]

    perturbed = _perturb_history_events(events, "window_swap", case=case, seed=2026)

    assert [event.item_id for event in perturbed] == [1, 2, 3, 4]
    assert [event.timestamp for event in perturbed] == [95 * day, 98 * day, 10 * day, 20 * day]
    assert all(event.timestamp is not None and event.timestamp < case.timestamp for event in perturbed)


def test_window_swap_is_noop_without_both_time_windows():
    day = 24 * 60 * 60
    case = EvaluationCase(user_id=7, item_id=99, timestamp=100 * day, event_id=10)
    events = [
        HistoryEvent(user_id=7, item_id=1, timestamp=96 * day, event_id=1),
        HistoryEvent(user_id=7, item_id=2, timestamp=98 * day, event_id=2),
    ]

    perturbed = _perturb_history_events(events, "window_swap", case=case, seed=2026)

    assert perturbed == events


def test_history_perturbation_cli_writes_required_outputs(tmp_path: Path, capsys):
    dataset = _diagnostic_dataset(tmp_path)
    output = tmp_path / "run"

    assert (
        main(
            [
                "evaluate",
                "history-perturbations",
                "--dataset-dir",
                str(dataset),
                "--output-dir",
                str(output),
                "--ks",
                "1",
                "--item-knn-max-history-items",
                "1",
                "--seed",
                "2026",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()

    assert "item_knn/order_reversal:" in captured.out
    for name in [
        "config.yaml",
        "metrics.json",
        "metrics_by_perturbation.csv",
        "metrics_delta.csv",
        "metrics_by_case.csv",
        "metrics_by_segment.csv",
        "command.txt",
        "git_commit.txt",
        "stdout.log",
        "stderr.log",
        "environment.json",
    ]:
        assert (output / name).exists()

    first = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    second_output = tmp_path / "run_again"
    main(
        [
            "evaluate",
            "history-perturbations",
            "--dataset-dir",
            str(dataset),
            "--output-dir",
            str(second_output),
            "--ks",
            "1",
            "--item-knn-max-history-items",
            "1",
            "--seed",
            "2026",
        ]
    )
    second = json.loads((second_output / "metrics.json").read_text(encoding="utf-8"))
    assert first["baselines"] == second["baselines"]
    assert first["deltas"] == second["deltas"]
    assert (output / "metrics_by_case.csv").read_text(encoding="utf-8") == (
        second_output / "metrics_by_case.csv"
    ).read_text(encoding="utf-8")
