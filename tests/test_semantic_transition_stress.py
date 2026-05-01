import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from tglrec.cli import main
from tglrec.data import schema
from tglrec.eval.semantic_transition_stress import run_semantic_transition_stress


def _write_processed_dataset(
    root: Path,
    interactions: list[dict[str, object]],
    item_metadata: dict[int, dict[str, object]],
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
    pd.DataFrame(
        [
            {
                schema.ITEM_ID: item_id,
                schema.RAW_ITEM_ID: str(item_id),
                **metadata,
            }
            for item_id, metadata in sorted(item_metadata.items())
        ]
    ).to_csv(root / "items.csv", index=False)


def test_semantic_transition_stress_builds_hard_candidates_and_ranker_metrics(
    tmp_path: Path,
):
    dataset = tmp_path / "processed"
    _write_processed_dataset(
        dataset,
        [
            {
                schema.EVENT_ID: 0,
                schema.USER_ID: 1,
                schema.ITEM_ID: 10,
                schema.TIMESTAMP: 10,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 1,
                schema.USER_ID: 1,
                schema.ITEM_ID: 20,
                schema.TIMESTAMP: 20,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 2,
                schema.USER_ID: 2,
                schema.ITEM_ID: 10,
                schema.TIMESTAMP: 30,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 3,
                schema.USER_ID: 2,
                schema.ITEM_ID: 30,
                schema.TIMESTAMP: 40,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 4,
                schema.USER_ID: 0,
                schema.ITEM_ID: 10,
                schema.TIMESTAMP: 100,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 5,
                schema.USER_ID: 0,
                schema.ITEM_ID: 20,
                schema.TIMESTAMP: 200,
                schema.SPLIT_LOO: "test",
            },
            {
                schema.EVENT_ID: 6,
                schema.USER_ID: 3,
                schema.ITEM_ID: 10,
                schema.TIMESTAMP: 300,
                schema.SPLIT_LOO: "train",
            },
            {
                schema.EVENT_ID: 7,
                schema.USER_ID: 3,
                schema.ITEM_ID: 50,
                schema.TIMESTAMP: 400,
                schema.SPLIT_LOO: "train",
            },
        ],
        {
            10: {"title": "Camera starter", "genres": "photo gear"},
            20: {"title": "Tripod mount", "genres": "support"},
            30: {"title": "Running socks", "genres": "fitness"},
            40: {"title": "Camera lens", "genres": "photo accessory"},
            50: {"title": "Camera bag", "genres": "photo storage"},
        },
    )

    result = run_semantic_transition_stress(
        dataset_dir=dataset,
        output_dir=tmp_path / "run",
        ks=(1, 2),
        max_history_items=1,
        per_source_top_k=10,
        command="synthetic",
    )

    assert result.metrics["semantic_hard_negative_coverage"] == pytest.approx(1.0)
    assert result.metrics["target_transition_evidence_rate"] == pytest.approx(1.0)
    assert result.metrics["semantic_overlap_semantic_trap_rate"] == pytest.approx(1.0)
    assert result.metrics["tdig_transition_transition_win_rate"] == pytest.approx(1.0)
    with (result.output_dir / "metrics_by_case.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["semantic_negative_item_id"] == "40"
    assert rows[0]["transition_negative_item_id"] == "30"
    assert rows[0]["target_has_transition_evidence"] == "1"
    assert rows[0]["stress_case_type"] == "target_transition_with_semantic_negative"
    assert json.loads(rows[0]["semantic_negative_tokens_json"]) == ["camera", "photo"]
    for name in [
        "config.yaml",
        "metrics.json",
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
        assert (result.output_dir / name).exists()


def test_semantic_transition_stress_cli_is_deterministic(tmp_path: Path, capsys):
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
        {
            1: {"title": "Space opera starter", "genres": "cosmos"},
            2: {"title": "Cooking memoir", "genres": "food"},
            3: {"title": "Space opera sequel", "genres": "cosmos"},
        },
    )
    first = tmp_path / "first"
    second = tmp_path / "second"

    assert (
        main(
            [
                "evaluate",
                "semantic-transition-stress",
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
                "5",
                "--max-eval-cases",
                "1",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "wrote semantic-transition stress run:" in captured.out
    run_semantic_transition_stress(
        dataset_dir=dataset,
        output_dir=second,
        ks=(1, 2),
        max_history_items=1,
        per_source_top_k=5,
        command="synthetic",
    )

    assert (first / "metrics_by_case.csv").read_text(encoding="utf-8") == (
        second / "metrics_by_case.csv"
    ).read_text(encoding="utf-8")
    config = (first / "config.yaml").read_text(encoding="utf-8")
    assert "candidate_mode: semantic_transition_hard_candidates" in config
    assert "diagnostic_rankers:" in config
    assert "max_eval_cases: 1" in config
