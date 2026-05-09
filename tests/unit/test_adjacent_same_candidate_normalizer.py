from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from llm4rec.data.adjacent_same_candidate import (
    canonical_domain_name,
    normalize_adjacent_same_candidate_tasks,
)
from llm4rec.data.week8_same_candidate import import_week8_same_candidate_task
from llm4rec.io.artifacts import read_jsonl


def test_canonical_domain_name_handles_adjacent_aliases() -> None:
    assert canonical_domain_name("amazon_beauty") == "beauty"
    assert canonical_domain_name("electronic") == "electronics"
    assert canonical_domain_name("amazon_electronics_small") == "electronics"
    assert canonical_domain_name("movie") == "movies"
    assert canonical_domain_name("amazon_movies_small") == "movies"
    with pytest.raises(ValueError, match="Unsupported"):
        canonical_domain_name("amazon_unknown")


def test_normalizer_preserves_candidate_order_and_imports(tmp_path: Path) -> None:
    source = tmp_path / "processed_4domains"
    domain_dir = source / "amazon_electronics_small"
    domain_dir.mkdir(parents=True)
    _write_adjacent_domain(domain_dir, split="test")

    tasks = normalize_adjacent_same_candidate_tasks(
        source_root=source,
        output_root=tmp_path / "staging",
        protocol_version="protocol_adjacent_test",
        domains=["electronic"],
        splits=["test"],
    )

    assert len(tasks) == 1
    staged = tasks[0].output_dir
    ranking = read_jsonl(staged / "ranking_test.jsonl")
    assert ranking[0]["candidate_items"] == ["i3", "i1", "i2"]
    assert ranking[0]["target_item"] == "i1"
    assert ranking[0]["source_event_id"] == "src-1"

    result = import_week8_same_candidate_task(
        staged,
        output_root=tmp_path / "artifacts",
        protocol_version="protocol_adjacent_test",
    )
    candidates = read_jsonl(result.output_dir / "candidates.jsonl")
    assert candidates[0]["candidate_items"] == ["i3", "i1", "i2"]
    assert candidates[0]["source_event_id"] == "src-1"
    assert candidates[0]["domain"] == "electronics"


def test_normalizer_rejects_multiple_positive_labels(tmp_path: Path) -> None:
    source = tmp_path / "processed_4domains"
    domain_dir = source / "amazon_movies_small"
    domain_dir.mkdir(parents=True)
    _write_adjacent_domain(domain_dir, split="valid", labels=[1, 1, 0], positive_item_id="")

    with pytest.raises(ValueError, match="Expected exactly one positive"):
        normalize_adjacent_same_candidate_tasks(
            source_root=source,
            output_root=tmp_path / "staging",
            protocol_version="protocol_adjacent_test",
            domains=["movies"],
            splits=["valid"],
        )


def _write_adjacent_domain(
    domain_dir: Path,
    *,
    split: str,
    labels: list[int] | None = None,
    positive_item_id: str = "i1",
) -> None:
    labels = labels or [0, 1, 0]
    row = {
        "candidate_item_ids": ["i3", "i1", "i2"],
        "candidate_labels": labels,
        "history": ["i0"],
        "positive_item_id": positive_item_id,
        "source_event_id": "src-1",
        "split_name": split,
        "timestamp": 123.0,
        "user_id": "u1",
    }
    (domain_dir / f"ranking_{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (domain_dir / "interactions.csv").write_text(
        "user_id,item_id,rating,timestamp\nu1,i0,1,1\n",
        encoding="utf-8",
    )
    with (domain_dir / "items.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "title", "categories", "candidate_text"])
        writer.writeheader()
        writer.writerow({"candidate_text": "Title: zero", "categories": "c", "item_id": "i0", "title": "zero"})
        writer.writerow({"candidate_text": "Title: one", "categories": "c", "item_id": "i1", "title": "one"})
        writer.writerow({"candidate_text": "Title: two", "categories": "c", "item_id": "i2", "title": "two"})
        writer.writerow({"candidate_text": "Title: three", "categories": "c", "item_id": "i3", "title": "three"})

