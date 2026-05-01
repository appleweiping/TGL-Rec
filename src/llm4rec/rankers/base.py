"""Unified ranker contracts for Phase 2A baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class RankingExample:
    """One user-target candidate-ranking example."""

    user_id: str
    history: list[str]
    target_item: str
    candidate_items: list[str]
    domain: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RankingResult:
    """Ranked candidates with scores and optional raw model output."""

    user_id: str
    items: list[str]
    scores: list[float]
    raw_output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRanker(Protocol):
    """Minimal contract shared by all Phase 2A rankers."""

    name: str

    def fit(
        self,
        train_interactions: list[dict[str, Any]],
        item_records: list[dict[str, Any]],
    ) -> None:
        """Fit ranker state from train-only interactions."""

    def rank(self, example: RankingExample) -> RankingResult:
        """Rank one candidate set."""

    def save_artifact(self, output_dir: str | Path) -> None:
        """Persist optional model artifacts."""


def result_from_scores(
    *,
    example: RankingExample,
    scores_by_item: dict[str, float],
    metadata: dict[str, Any] | None = None,
) -> RankingResult:
    """Sort candidates by score descending with deterministic item-id tie-breaks."""

    ordered = sorted(
        [str(item_id) for item_id in example.candidate_items],
        key=lambda item_id: (-float(scores_by_item.get(item_id, 0.0)), item_id),
    )
    return RankingResult(
        user_id=example.user_id,
        items=ordered,
        scores=[float(scores_by_item.get(item_id, 0.0)) for item_id in ordered],
        raw_output=None,
        metadata=metadata or {},
    )
