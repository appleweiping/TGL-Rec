from __future__ import annotations

from llm4rec.data.sequence_transforms import (
    apply_sequence_transform,
    popularity_sorted_sequence,
    recent_k_sequence,
    remove_recent_k_sequence,
    reversed_sequence,
    shuffled_sequence,
)


def test_sequence_transforms_are_deterministic() -> None:
    history = ["i1", "i2", "i3", "i4"]
    assert reversed_sequence(history) == ["i4", "i3", "i2", "i1"]
    assert shuffled_sequence(history, seed=7) == shuffled_sequence(history, seed=7)
    assert recent_k_sequence(history, k=2) == ["i3", "i4"]
    assert remove_recent_k_sequence(history, k=2) == ["i1", "i2"]


def test_popularity_sorted_transform() -> None:
    assert popularity_sorted_sequence(["i3", "i2", "i1"], {"i1": 2, "i2": 2, "i3": 5}) == [
        "i3",
        "i1",
        "i2",
    ]


def test_apply_sequence_transform_dispatch() -> None:
    assert apply_sequence_transform(["i1", "i2"], transform="original") == ["i1", "i2"]
