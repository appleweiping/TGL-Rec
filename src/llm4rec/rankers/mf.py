"""Dependency-free BPR matrix-factorization ranker for tiny smoke runs."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from llm4rec.io.artifacts import write_json
from llm4rec.rankers.base import RankingExample, RankingResult, result_from_scores


class MatrixFactorizationRanker:
    """Small CPU BPR-MF baseline with deterministic negative sampling."""

    name = "mf"

    def __init__(
        self,
        *,
        factors: int = 8,
        epochs: int = 20,
        learning_rate: float = 0.05,
        regularization: float = 0.002,
        seed: int = 0,
    ) -> None:
        self.factors = int(factors)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.seed = int(seed)
        self.user_factors: dict[str, list[float]] = {}
        self.item_factors: dict[str, list[float]] = {}
        self.item_popularity: Counter[str] = Counter()
        self.user_positives: dict[str, set[str]] = {}
        self.item_ids: list[str] = []

    def fit(
        self,
        train_interactions: list[dict[str, Any]],
        item_records: list[dict[str, Any]],
    ) -> None:
        rng = random.Random(self.seed)
        users = sorted({str(row["user_id"]) for row in train_interactions})
        self.item_ids = sorted({str(row["item_id"]) for row in item_records})
        self.item_popularity = Counter(str(row["item_id"]) for row in train_interactions)
        positives: dict[str, set[str]] = defaultdict(set)
        for row in train_interactions:
            positives[str(row["user_id"])].add(str(row["item_id"]))
        self.user_positives = {user_id: set(items) for user_id, items in positives.items()}
        self.user_factors = {user_id: self._random_vector(rng) for user_id in users}
        self.item_factors = {item_id: self._random_vector(rng) for item_id in self.item_ids}

        pairs = [(str(row["user_id"]), str(row["item_id"])) for row in train_interactions]
        for _epoch in range(self.epochs):
            rng.shuffle(pairs)
            for user_id, positive_item in pairs:
                negative_item = self._sample_negative(user_id, rng)
                if negative_item is None:
                    continue
                self._update(user_id, positive_item, negative_item)

    def rank(self, example: RankingExample) -> RankingResult:
        scores = {str(item_id): self._score(example.user_id, str(item_id)) for item_id in example.candidate_items}
        return result_from_scores(
            example=example,
            scores_by_item=scores,
            metadata={
                "epochs": self.epochs,
                "factors": self.factors,
                "loss": "bpr",
                "seed": self.seed,
            },
        )

    def save_artifact(self, output_dir: str | Path) -> None:
        write_json(
            Path(output_dir) / "mf_model_metadata.json",
            {
                "epochs": self.epochs,
                "factors": self.factors,
                "item_count": len(self.item_factors),
                "learning_rate": self.learning_rate,
                "regularization": self.regularization,
                "seed": self.seed,
                "user_count": len(self.user_factors),
            },
        )

    def _random_vector(self, rng: random.Random) -> list[float]:
        return [(rng.random() - 0.5) * 0.2 for _ in range(self.factors)]

    def _sample_negative(self, user_id: str, rng: random.Random) -> str | None:
        positives = self.user_positives.get(user_id, set())
        candidates = [item_id for item_id in self.item_ids if item_id not in positives]
        if not candidates:
            return None
        return candidates[rng.randrange(len(candidates))]

    def _score(self, user_id: str, item_id: str) -> float:
        user = self.user_factors.get(user_id)
        item = self.item_factors.get(item_id)
        if user is None or item is None:
            return float(self.item_popularity.get(item_id, 0)) * 1e-6
        return sum(u * i for u, i in zip(user, item))

    def _update(self, user_id: str, positive_item: str, negative_item: str) -> None:
        user = self.user_factors[user_id]
        positive = self.item_factors[positive_item]
        negative = self.item_factors[negative_item]
        diff = sum(u * (pi - ni) for u, pi, ni in zip(user, positive, negative))
        diff = max(min(diff, 35.0), -35.0)
        coefficient = 1.0 / (1.0 + math.exp(diff))
        old_user = list(user)
        lr = self.learning_rate
        reg = self.regularization
        for idx in range(self.factors):
            user[idx] += lr * (coefficient * (positive[idx] - negative[idx]) - reg * user[idx])
            positive[idx] += lr * (coefficient * old_user[idx] - reg * positive[idx])
            negative[idx] += lr * (-coefficient * old_user[idx] - reg * negative[idx])
