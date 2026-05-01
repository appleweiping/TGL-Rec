"""Sequential ranker interfaces and lightweight smoke baseline."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from llm4rec.io.artifacts import write_json
from llm4rec.rankers.base import RankingExample, RankingResult, result_from_scores


class MarkovTransitionRanker:
    """First-order next-item transition baseline for smoke diagnostics.

    This is a real deterministic sequential baseline, but it is marked for smoke/pre-experiment
    use rather than as a formal SASRec/GRU4Rec replacement.
    """

    name = "markov_transition"

    def __init__(self) -> None:
        self.transition_counts: Counter[tuple[str, str]] = Counter()
        self.item_popularity: Counter[str] = Counter()
        self.item_ids: list[str] = []

    def fit(self, train_interactions: list[dict[str, Any]], item_records: list[dict[str, Any]]) -> None:
        self.item_ids = sorted(str(row["item_id"]) for row in item_records)
        self.item_popularity = Counter(str(row["item_id"]) for row in train_interactions)
        by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in train_interactions:
            by_user[str(row["user_id"])].append(row)
        for rows in by_user.values():
            ordered = sorted(
                rows,
                key=lambda row: (
                    float(row["timestamp"]) if row.get("timestamp") is not None else -1.0,
                    str(row["item_id"]),
                ),
            )
            for left, right in zip(ordered, ordered[1:]):
                self.transition_counts[(str(left["item_id"]), str(right["item_id"]))] += 1

    def rank(self, example: RankingExample) -> RankingResult:
        last_item = str(example.history[-1]) if example.history else None
        scores: dict[str, float] = {}
        for item_id in example.candidate_items:
            item = str(item_id)
            transition = 0.0 if last_item is None else float(self.transition_counts.get((last_item, item), 0))
            popularity = float(self.item_popularity.get(item, 0)) * 1e-6
            scores[item] = transition + popularity
        return result_from_scores(
            example=example,
            scores_by_item=scores,
            metadata={
                "last_history_item": last_item,
                "reportable": False,
                "sequential_baseline": "first_order_markov_transition",
            },
        )

    def save_artifact(self, output_dir: str | Path) -> None:
        write_json(
            Path(output_dir) / "markov_transition_metadata.json",
            {
                "item_count": len(self.item_ids),
                "nonzero_transition_count": len(self.transition_counts),
                "reportable": False,
            },
        )


class SequentialModelInterface:
    """Interface placeholder for formal sequential models not yet implemented locally."""

    def __init__(self, *, model_name: str, reportable: bool = False) -> None:
        self.name = model_name
        self.reportable = bool(reportable)

    def fit(self, train_interactions: list[dict[str, Any]], item_records: list[dict[str, Any]]) -> None:
        raise NotImplementedError(
            f"{self.name} training is not implemented in Phase 4. "
            "Use an external validated wrapper or implement the formal model before reportable runs."
        )

    def rank(self, example: RankingExample) -> RankingResult:
        raise NotImplementedError(f"{self.name} ranking is not implemented in Phase 4.")

    def save_artifact(self, output_dir: str | Path) -> None:
        write_json(
            Path(output_dir) / f"{self.name}_interface.json",
            {"implemented": False, "reportable": self.reportable},
        )


class SASRecInterface(SequentialModelInterface):
    """SASRec formal-baseline interface placeholder."""

    def __init__(self, *, reportable: bool = False) -> None:
        super().__init__(model_name="sasrec", reportable=reportable)


class GRU4RecInterface(SequentialModelInterface):
    """GRU4Rec formal-baseline interface placeholder."""

    def __init__(self, *, reportable: bool = False) -> None:
        super().__init__(model_name="gru4rec", reportable=reportable)
