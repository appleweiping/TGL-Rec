import json
from pathlib import Path

import pandas as pd
import pytest

from tglrec.cli import main
from tglrec.data import schema
from tglrec.graph.tdig import (
    DAY_SECONDS,
    MONTH_SECONDS,
    SAME_SESSION_SECONDS,
    WEEK_SECONDS,
    build_tdig_from_events,
    build_tdig_from_processed_split,
    gap_bucket,
)


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


def test_gap_bucket_boundaries():
    assert gap_bucket(0) == "same_session"
    assert gap_bucket(SAME_SESSION_SECONDS) == "same_session"
    assert gap_bucket(SAME_SESSION_SECONDS + 1) == "within_1d"
    assert gap_bucket(DAY_SECONDS) == "within_1d"
    assert gap_bucket(DAY_SECONDS + 1) == "within_1w"
    assert gap_bucket(WEEK_SECONDS) == "within_1w"
    assert gap_bucket(WEEK_SECONDS + 1) == "within_1m"
    assert gap_bucket(MONTH_SECONDS) == "within_1m"
    assert gap_bucket(MONTH_SECONDS + 1) == "long_gap"
    with pytest.raises(ValueError, match="non-negative"):
        gap_bucket(-1)


def test_tdig_direct_edges_are_directional_and_compute_asymmetry():
    day = DAY_SECONDS
    events = pd.DataFrame(
        [
            {schema.EVENT_ID: 0, schema.USER_ID: 0, schema.ITEM_ID: 1, schema.TIMESTAMP: 0},
            {schema.EVENT_ID: 1, schema.USER_ID: 0, schema.ITEM_ID: 2, schema.TIMESTAMP: day},
            {schema.EVENT_ID: 2, schema.USER_ID: 1, schema.ITEM_ID: 1, schema.TIMESTAMP: 10 * day},
            {schema.EVENT_ID: 3, schema.USER_ID: 1, schema.ITEM_ID: 2, schema.TIMESTAMP: 11 * day},
            {schema.EVENT_ID: 4, schema.USER_ID: 2, schema.ITEM_ID: 2, schema.TIMESTAMP: 20 * day},
            {schema.EVENT_ID: 5, schema.USER_ID: 2, schema.ITEM_ID: 1, schema.TIMESTAMP: 21 * day},
        ]
    )

    graph, metadata = build_tdig_from_events(events)

    forward = graph.edges[(1, 2)]
    reverse = graph.edges[(2, 1)]
    assert metadata["num_transitions"] == 3
    assert forward.support == 2
    assert reverse.support == 1
    assert forward.direction_asymmetry == pytest.approx(1 / 3)
    assert reverse.direction_asymmetry == pytest.approx(-1 / 3)
    assert forward.gap_histogram["within_1d"] == 2
    assert forward.transition_probability == pytest.approx(1.0)
    assert forward.lift == pytest.approx(1.5)
    assert forward.pmi == pytest.approx(0.4054651081081644)


def test_tdig_processed_split_uses_train_only_and_does_not_leak_val_or_test(tmp_path: Path):
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
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 2,
                schema.USER_ID: 0,
                schema.ITEM_ID: 9,
                schema.TIMESTAMP: 300,
                schema.SPLIT_LOO: "val",
            },
            {
                schema.EVENT_ID: 3,
                schema.USER_ID: 0,
                schema.ITEM_ID: 10,
                schema.TIMESTAMP: 400,
                schema.SPLIT_LOO: "test",
            },
            {
                schema.EVENT_ID: 4,
                schema.USER_ID: 1,
                schema.ITEM_ID: 9,
                schema.TIMESTAMP: 500,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 5,
                schema.USER_ID: 1,
                schema.ITEM_ID: 10,
                schema.TIMESTAMP: 600,
                schema.SPLIT_LOO: "train",
            },
        ],
    )

    graph, metadata = build_tdig_from_processed_split(dataset_dir=dataset)

    assert (1, 2) in graph.edges
    assert (2, 9) not in graph.edges
    assert (9, 10) in graph.edges
    assert metadata["used_event_count"] == 4
    assert metadata["input_event_count"] == 6
    assert metadata["split_column"] == schema.SPLIT_LOO


def test_tdig_processed_split_supports_strict_as_of_train_evidence(tmp_path: Path):
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
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 2,
                schema.USER_ID: 1,
                schema.ITEM_ID: 9,
                schema.TIMESTAMP: 500,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 3,
                schema.USER_ID: 1,
                schema.ITEM_ID: 10,
                schema.TIMESTAMP: 600,
                schema.SPLIT_LOO: "train",
            },
        ],
    )

    graph, metadata = build_tdig_from_processed_split(
        dataset_dir=dataset,
        strict_before_timestamp=300,
    )

    assert (1, 2) in graph.edges
    assert (9, 10) not in graph.edges
    assert metadata["used_event_count"] == 2
    assert metadata["strict_before_timestamp"] == 300


def test_tdig_retrieval_is_deterministic_with_stable_tie_breaks():
    events = pd.DataFrame(
        [
            {schema.EVENT_ID: 0, schema.USER_ID: 0, schema.ITEM_ID: 1, schema.TIMESTAMP: 100},
            {schema.EVENT_ID: 1, schema.USER_ID: 0, schema.ITEM_ID: 3, schema.TIMESTAMP: 200},
            {schema.EVENT_ID: 2, schema.USER_ID: 1, schema.ITEM_ID: 1, schema.TIMESTAMP: 100},
            {schema.EVENT_ID: 3, schema.USER_ID: 1, schema.ITEM_ID: 2, schema.TIMESTAMP: 200},
        ]
    )

    first_graph, _ = build_tdig_from_events(events)
    second_graph, _ = build_tdig_from_events(events.sample(frac=1.0, random_state=17))

    first = [candidate.to_dict() for candidate in first_graph.retrieve_direct(1, top_k=10)]
    second = [candidate.to_dict() for candidate in second_graph.retrieve_direct(1, top_k=10)]
    assert [candidate["target_item_id"] for candidate in first] == [2, 3]
    assert first == second


def test_tdig_rejects_same_timestamp_ties_without_event_ids():
    events = pd.DataFrame(
        [
            {schema.USER_ID: 0, schema.ITEM_ID: 1, schema.TIMESTAMP: 100},
            {schema.USER_ID: 0, schema.ITEM_ID: 2, schema.TIMESTAMP: 100},
        ]
    )

    with pytest.raises(ValueError, match="same-user same-timestamp ties"):
        build_tdig_from_events(events)


def test_tdig_skips_same_timestamp_transitions_by_default():
    events = pd.DataFrame(
        [
            {schema.EVENT_ID: 0, schema.USER_ID: 0, schema.ITEM_ID: 1, schema.TIMESTAMP: 100},
            {schema.EVENT_ID: 1, schema.USER_ID: 0, schema.ITEM_ID: 2, schema.TIMESTAMP: 100},
            {schema.EVENT_ID: 2, schema.USER_ID: 0, schema.ITEM_ID: 3, schema.TIMESTAMP: 200},
        ]
    )

    graph, metadata = build_tdig_from_events(events)
    graph_with_ties, metadata_with_ties = build_tdig_from_events(
        events,
        include_same_timestamp_transitions=True,
    )

    assert (1, 2) not in graph.edges
    assert (2, 3) in graph.edges
    assert metadata["skipped_same_timestamp_transitions"] == 1
    assert (1, 2) in graph_with_ties.edges
    assert metadata_with_ties["skipped_same_timestamp_transitions"] == 0


def test_tdig_gap_bucket_retrieval_returns_only_bucket_evidence():
    events = pd.DataFrame(
        [
            {schema.EVENT_ID: 0, schema.USER_ID: 0, schema.ITEM_ID: 1, schema.TIMESTAMP: 0},
            {schema.EVENT_ID: 1, schema.USER_ID: 0, schema.ITEM_ID: 2, schema.TIMESTAMP: 60},
            {schema.EVENT_ID: 2, schema.USER_ID: 1, schema.ITEM_ID: 1, schema.TIMESTAMP: 0},
            {
                schema.EVENT_ID: 3,
                schema.USER_ID: 1,
                schema.ITEM_ID: 3,
                schema.TIMESTAMP: MONTH_SECONDS + 1,
            },
        ]
    )

    graph, _ = build_tdig_from_events(events)

    same_session_targets = [
        candidate.target_item_id
        for candidate in graph.retrieve_direct(1, top_k=10, gap_bucket="same_session")
    ]
    long_gap_targets = [
        candidate.target_item_id
        for candidate in graph.retrieve_direct(1, top_k=10, gap_bucket="long_gap")
    ]

    assert same_session_targets == [2]
    assert long_gap_targets == [3]


def test_tdig_cli_writes_artifact_files(tmp_path: Path, capsys):
    dataset = tmp_path / "processed"
    output = tmp_path / "tdig"
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
                schema.SPLIT_LOO: "train",
            },
        ],
    )

    assert (
        main(
            [
                "graph",
                "build-tdig",
                "--dataset-dir",
                str(dataset),
                "--output-dir",
                str(output),
                "--strict-before-timestamp",
                "250",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()

    assert "wrote TDIG artifact:" in captured.out
    assert "edges=1 transitions=1" in captured.out
    for name in [
        "edges.csv",
        "metadata.json",
        "config.yaml",
        "command.txt",
        "git_commit.txt",
        "created_at_utc.txt",
        "stdout.log",
        "stderr.log",
        "environment.json",
        "checksums.json",
    ]:
        assert (output / name).exists()
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["num_edges"] == 1
    assert metadata["strict_before_timestamp"] == 250
    assert metadata["include_same_timestamp_transitions"] is False
    assert metadata["dataset_provenance"]["interactions_csv"]["sha256"]
    assert "warnings" in metadata["dataset_provenance"]
    assert metadata["leakage_policy"].startswith("Only events matching")
