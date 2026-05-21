"""Learned need-gate for TGL-Rec reportable scoring.

The gate decides how much to trust temporal evidence vs semantic similarity
for each (user, candidate) pair. It is intentionally lightweight (26-51 params)
to remain interpretable and avoid overfitting.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GateConfig:
    """Configuration for the learned need-gate."""

    input_dim: int = 25
    use_interaction: bool = True
    learning_rate: float = 0.01
    l2_reg: float = 0.001
    max_epochs: int = 100
    patience: int = 10
    batch_size: int = 256

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GateConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class GateWeights:
    """Learned parameters of the need-gate."""

    w: list[float] = field(default_factory=list)
    b: float = 0.0
    trained: bool = False
    train_metrics: dict[str, float] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "w": self.w,
            "b": self.b,
            "trained": self.trained,
            "train_metrics": self.train_metrics,
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> "GateWeights":
        data = json.loads(path.read_text())
        return cls(
            w=data["w"],
            b=data["b"],
            trained=data.get("trained", True),
            train_metrics=data.get("train_metrics", {}),
        )


class LearnedNeedGate:
    """Lightweight logistic gate: σ(w·[n_u; e_c; n_u⊙e_c] + b).

    Predicts α ∈ [0,1] where high α means "trust temporal evidence."
    """

    def __init__(self, config: GateConfig | None = None) -> None:
        self.config = config or GateConfig()
        self.weights = GateWeights()

    def predict(self, need_state: list[float], evidence_features: list[float]) -> float:
        if not self.weights.trained:
            return self._deterministic_fallback(need_state, evidence_features)
        x = self._build_input(need_state, evidence_features)
        logit = sum(wi * xi for wi, xi in zip(self.weights.w, x)) + self.weights.b
        return _sigmoid(logit)

    def train(
        self,
        examples: list[dict[str, Any]],
        valid_examples: list[dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        """Train the gate via mini-batch SGD with early stopping.

        Each example: {"need_state": [...], "evidence_features": [...], "label": 0/1, "weight": float}
        """
        if not examples:
            return {"error": "no training examples"}

        dim = len(self._build_input(examples[0]["need_state"], examples[0]["evidence_features"]))
        self.weights.w = [0.0] * dim
        self.weights.b = 0.0

        lr = self.config.learning_rate
        reg = self.config.l2_reg
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.max_epochs):
            epoch_loss = 0.0
            n = 0
            for i in range(0, len(examples), self.config.batch_size):
                batch = examples[i : i + self.config.batch_size]
                grad_w = [0.0] * dim
                grad_b = 0.0
                batch_loss = 0.0

                for ex in batch:
                    x = self._build_input(ex["need_state"], ex["evidence_features"])
                    y = float(ex["label"])
                    sample_weight = float(ex.get("weight", 1.0))
                    pred = _sigmoid(sum(wi * xi for wi, xi in zip(self.weights.w, x)) + self.weights.b)
                    error = (pred - y) * sample_weight
                    for j in range(dim):
                        grad_w[j] += error * x[j]
                    grad_b += error
                    loss = -sample_weight * (y * math.log(max(pred, 1e-10)) + (1 - y) * math.log(max(1 - pred, 1e-10)))
                    batch_loss += loss

                bs = len(batch)
                for j in range(dim):
                    self.weights.w[j] -= lr * (grad_w[j] / bs + reg * self.weights.w[j])
                self.weights.b -= lr * grad_b / bs
                epoch_loss += batch_loss
                n += bs

            epoch_loss /= max(n, 1)

            if valid_examples:
                val_loss = self._compute_loss(valid_examples)
            else:
                val_loss = epoch_loss

            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    break

        self.weights.trained = True
        self.weights.train_metrics = {
            "final_train_loss": float(epoch_loss),
            "best_val_loss": float(best_loss),
            "epochs_trained": epoch + 1,
            "dim": dim,
        }
        return self.weights.train_metrics

    def save(self, path: Path) -> None:
        self.weights.save(path)

    def load(self, path: Path) -> None:
        self.weights = GateWeights.load(path)

    def _build_input(self, need_state: list[float], evidence_features: list[float]) -> list[float]:
        ns = need_state[:5] if len(need_state) >= 5 else need_state + [0.0] * (5 - len(need_state))
        ef = evidence_features[:10] if len(evidence_features) >= 10 else evidence_features + [0.0] * (10 - len(evidence_features))
        x = ns + ef
        if self.config.use_interaction:
            interaction = [ns[i] * ef[i] for i in range(min(5, len(ef)))]
            x += interaction + [0.0] * (5 - len(interaction))
        return x

    def _deterministic_fallback(self, need_state: list[float], evidence_features: list[float]) -> float:
        temporal_signal = sum(evidence_features[:6]) if len(evidence_features) >= 6 else sum(evidence_features)
        semantic_signal = evidence_features[6] if len(evidence_features) > 6 else 0.0
        logit = temporal_signal - semantic_signal
        return _sigmoid(logit)

    def _compute_loss(self, examples: list[dict[str, Any]]) -> float:
        total = 0.0
        for ex in examples:
            x = self._build_input(ex["need_state"], ex["evidence_features"])
            y = float(ex["label"])
            w = float(ex.get("weight", 1.0))
            pred = _sigmoid(sum(wi * xi for wi, xi in zip(self.weights.w, x)) + self.weights.b)
            total -= w * (y * math.log(max(pred, 1e-10)) + (1 - y) * math.log(max(1 - pred, 1e-10)))
        return total / max(len(examples), 1)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
