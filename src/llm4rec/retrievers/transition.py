"""Transition-graph retriever."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from llm4rec.graph.transition_graph import build_transition_edges
from llm4rec.retrievers.base import RetrievalResult


class TransitionRetriever:
    """Retrieve items that commonly follow recent history items."""

    name = "transition"

    def __init__(self, *, max_history_items: int = 5) -> None:
        self.max_history_items = int(max_history_items)
        self.transition_scores: dict[str, dict[str, float]] = {}

    def fit(
        self,
        train_interactions: list[dict[str, Any]],
        item_records: list[dict[str, Any]],
    ) -> None:
        scores: dict[str, dict[str, float]] = defaultdict(dict)
        for edge in build_transition_edges(train_interactions):
            source = str(edge["source_item"])
            target = str(edge["target_item"])
            scores[source][target] = float(edge["count"])
        self.transition_scores = {source: dict(targets) for source, targets in scores.items()}

    def retrieve(
        self,
        *,
        user_id: str,
        history: list[str],
        top_k: int,
        domain: str | None = None,
    ) -> RetrievalResult:
        scores: dict[str, float] = defaultdict(float)
        for source in history[-self.max_history_items :]:
            for target, score in self.transition_scores.get(str(source), {}).items():
                if target in history:
                    continue
                scores[target] += score
        ordered = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))[:top_k]
        return RetrievalResult(
            user_id=user_id,
            items=ordered,
            scores=[float(scores[item_id]) for item_id in ordered],
            metadata={"max_history_items": self.max_history_items, "retriever": self.name},
        )

    def save_artifact(self, output_dir: str | Path) -> None:
        return None
