"""Vectorized TimeGraphEvidenceRec scorer for compact shared-pool candidates."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from llm4rec.scoring.candidate_batch import CandidateBatch


class TimeGraphEvidenceSharedPoolScorer:
    """Score evidence candidates without graph traversal for every candidate."""

    name = "time_graph_evidence_shared_pool"

    def __init__(
        self,
        *,
        transition_counts: Counter[tuple[str, str]],
        category_by_item: dict[str, str],
        dynamic_scorer: Any | None = None,
        dynamic_weight: float = 0.25,
        candidate_pool_items: list[str] | None = None,
        negative_pool_items: list[str] | None = None,
    ) -> None:
        self.transition_counts = transition_counts
        self.category_by_item = {str(key): str(value) for key, value in category_by_item.items()}
        self.dynamic_scorer = dynamic_scorer
        self.dynamic_weight = float(dynamic_weight)
        self.transitions_by_source: dict[str, dict[str, float]] = defaultdict(dict)
        for (source, target), count in transition_counts.items():
            self.transitions_by_source[str(source)][str(target)] = float(count)
        self.candidate_pool_items = [str(item) for item in candidate_pool_items or []]
        self.negative_pool_items = [str(item) for item in negative_pool_items or []]
        self.pool_index = {item: index for index, item in enumerate(self.candidate_pool_items)}
        self.negative_index = {item: index for index, item in enumerate(self.negative_pool_items)}
        self.pool_category_indices = self._category_indices(self.candidate_pool_items)
        self.negative_category_indices = self._category_indices(self.negative_pool_items)
        self.metadata = {
            "dynamic_encoder_enabled": dynamic_scorer is not None,
            "evidence_constructed_from": "train_only",
            "save_top_evidence_only": True,
            "scorer": self.name,
            "transition_edges": len(transition_counts),
            "vectorized": True,
        }

    @classmethod
    def from_state(
        cls,
        state: dict[str, Any],
        *,
        dynamic_scorer: Any | None = None,
        candidate_pool_items: list[str] | None = None,
        negative_pool_items: list[str] | None = None,
    ) -> "TimeGraphEvidenceSharedPoolScorer":
        """Build from the paper-matrix evidence state."""

        return cls(
            transition_counts=state["transition_counts"],
            category_by_item=state["category_by_item"],
            dynamic_scorer=dynamic_scorer,
            candidate_pool_items=candidate_pool_items,
            negative_pool_items=negative_pool_items,
        )

    def score_batch(self, batch: CandidateBatch) -> Any:
        import numpy as np

        scores = np.zeros((batch.batch_size, batch.candidate_size), dtype=np.float32)
        for row_index, (history, candidates, target) in enumerate(
            zip(batch.histories, batch.candidate_item_ids, batch.target_items)
        ):
            recent = list(reversed(history[-3:]))
            total_constant = sum(0.05 / float(rank) for rank, _source in enumerate(recent, start=1))
            if total_constant:
                scores[row_index, :] = total_constant
            index, category_indices = self._row_indices(candidates)
            for rank, source in enumerate(recent, start=1):
                source = str(source)
                source_pos = index.get(source)
                if source_pos is not None:
                    scores[row_index, source_pos] -= 0.05 / float(rank)
                for target_item, count in self.transitions_by_source.get(source, {}).items():
                    pos = index.get(target_item)
                    if pos is not None:
                        scores[row_index, pos] += float(count)
            dominant = self._dominant_recent_category(recent)
            if dominant:
                for pos in category_indices.get(dominant, []):
                    scores[row_index, pos] += 0.1
                if str(target) not in index and self.category_by_item.get(str(target), "") == dominant:
                    scores[row_index, -1] += 0.1
        if self.dynamic_scorer is not None:
            scores = scores + self.dynamic_weight * self.dynamic_scorer.score_batch(batch)
        return scores

    def metadata_for_top_items(
        self,
        *,
        top_items: list[str],
        top_scores: list[float],
        limit: int = 100,
    ) -> dict[str, Any]:
        """Return compact evidence metadata only for saved top predictions."""

        capped_items = top_items[: int(limit)]
        capped_scores = top_scores[: int(limit)]
        return {
            "top_evidence": {
                str(item): {"score": float(score)}
                for item, score in zip(capped_items, capped_scores)
            },
            "top_evidence_count": len(capped_items),
        }

    def _row_indices(self, candidates: list[str]) -> tuple[dict[str, int], dict[str, list[int]]]:
        if self.candidate_pool_items and candidates is self.candidate_pool_items:
            return self.pool_index, self.pool_category_indices
        if self.candidate_pool_items and candidates == self.candidate_pool_items:
            return self.pool_index, self.pool_category_indices
        if self.negative_pool_items and len(candidates) == len(self.negative_pool_items) + 1 and candidates[:-1] == self.negative_pool_items:
            index = dict(self.negative_index)
            index[str(candidates[-1])] = len(candidates) - 1
            category_indices = {key: list(value) for key, value in self.negative_category_indices.items()}
            category = self.category_by_item.get(str(candidates[-1]), "")
            if category:
                category_indices.setdefault(category, []).append(len(candidates) - 1)
            return index, category_indices
        index = {str(item): pos for pos, item in enumerate(candidates)}
        return index, self._category_indices(candidates)

    def _dominant_recent_category(self, recent: list[str]) -> str:
        categories = [self.category_by_item.get(str(item), "") for item in recent]
        categories = [category for category in categories if category]
        return Counter(categories).most_common(1)[0][0] if categories else ""

    def _category_indices(self, items: list[str]) -> dict[str, list[int]]:
        output: dict[str, list[int]] = defaultdict(list)
        for index, item in enumerate(items):
            category = self.category_by_item.get(str(item), "")
            if category:
                output[category].append(index)
        return dict(output)
