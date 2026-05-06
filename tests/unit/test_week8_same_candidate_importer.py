from __future__ import annotations

import json
from pathlib import Path

from llm4rec.data.week8_same_candidate import import_week8_same_candidate_task
from llm4rec.io.artifacts import read_jsonl


def test_week8_same_candidate_importer_preserves_event_candidates(tmp_path: Path) -> None:
    task_dir = tmp_path / "books_large10000_100neg_test_same_candidate"
    task_dir.mkdir()
    (task_dir / "ranking_test.jsonl").write_text(
        json.dumps(
            {
                "event_id": "e1",
                "history": ["i0", "i00"],
                "positive_item": "i2",
                "split": "test",
                "user_id": "u1",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (task_dir / "candidate_items.csv").write_text(
        "\n".join(
            [
                "event_id,user_id,item_id,label",
                "e1,u1,i1,0",
                "e1,u1,i2,1",
                "e1,u1,i3,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (task_dir / "train_interactions.csv").write_text(
        "user_id,item_id,timestamp,rating\nu1,i0,1,1\nu1,i00,2,1\n",
        encoding="utf-8",
    )
    (task_dir / "item_metadata.csv").write_text(
        "item_id,title,category\ni1,Item 1,books\ni2,Item 2,books\ni3,Item 3,books\n",
        encoding="utf-8",
    )
    (task_dir / "metadata.json").write_text('{"seed": 20260506}\n', encoding="utf-8")

    result = import_week8_same_candidate_task(
        task_dir,
        output_root=tmp_path / "artifacts",
        protocol_version="protocol_test_week8",
    )

    candidates = read_jsonl(result.output_dir / "candidates.jsonl")
    splits = read_jsonl(result.output_dir / "splits.jsonl")
    metadata = read_jsonl(result.output_dir / "item_metadata.jsonl")

    assert result.manifest["domain"] == "books"
    assert result.manifest["split"] == "test"
    assert candidates == [
        {
            "candidate_items": ["i1", "i2", "i3"],
            "candidate_source": "week8_same_candidate_external_task",
            "domain": "books",
            "event_id": "e1",
            "history": ["i0", "i00"],
            "protocol_version": "protocol_test_week8",
            "source_event_id": "e1",
            "split": "test",
            "target_item": "i2",
            "user_id": "u1",
        }
    ]
    assert [row["split"] for row in splits] == ["train", "train", "test"]
    assert metadata[1]["item_id"] == "i2"


def test_week8_same_candidate_importer_merges_valid_and_test(tmp_path: Path) -> None:
    valid_dir = _write_task(tmp_path, split="valid", event_id="ev", target="i2")
    test_dir = _write_task(tmp_path, split="test", event_id="et", target="i3")

    import_week8_same_candidate_task(
        valid_dir,
        output_root=tmp_path / "artifacts",
        protocol_version="protocol_test_week8",
    )
    result = import_week8_same_candidate_task(
        test_dir,
        output_root=tmp_path / "artifacts",
        protocol_version="protocol_test_week8",
    )

    candidates = read_jsonl(result.output_dir / "candidates.jsonl")
    splits = read_jsonl(result.output_dir / "splits.jsonl")

    assert [(row["event_id"], row["split"], row["target_item"]) for row in candidates] == [
        ("ev", "valid", "i2"),
        ("et", "test", "i3"),
    ]
    assert [row["split"] for row in splits] == ["valid", "train", "test"]


def _write_task(tmp_path: Path, *, split: str, event_id: str, target: str) -> Path:
    task_dir = tmp_path / f"books_large10000_100neg_{split}_same_candidate"
    task_dir.mkdir()
    (task_dir / f"ranking_{split}.jsonl").write_text(
        json.dumps(
            {
                "candidate_items": ["i1", "i2", "i3"],
                "event_id": event_id,
                "positive_item": target,
                "split": split,
                "user_id": "u1",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (task_dir / "candidate_items.csv").write_text(
        "event_id,user_id,candidate_items\n"
        f'{event_id},u1,"[""i1"", ""i2"", ""i3""]"\n',
        encoding="utf-8",
    )
    (task_dir / "train_interactions.csv").write_text(
        "user_id,item_id,timestamp,rating\nu1,i0,1,1\n",
        encoding="utf-8",
    )
    (task_dir / "item_metadata.csv").write_text(
        "item_id,title\ni1,Item 1\ni2,Item 2\ni3,Item 3\n",
        encoding="utf-8",
    )
    return task_dir
