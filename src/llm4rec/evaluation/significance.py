"""Significance-test interfaces for paired recommendation metrics."""

from __future__ import annotations

import random
from typing import Any


def paired_randomization_test(
    baseline_scores: list[float],
    treatment_scores: list[float],
    *,
    num_rounds: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Two-sided paired randomization test for per-case metric scores."""

    if len(baseline_scores) != len(treatment_scores):
        raise ValueError("baseline_scores and treatment_scores must have the same length")
    if len(baseline_scores) < 2:
        return {"p_value": None, "warning": "insufficient_sample", "n": len(baseline_scores)}
    observed = abs(_mean_delta(baseline_scores, treatment_scores))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(int(num_rounds)):
        left: list[float] = []
        right: list[float] = []
        for base, treatment in zip(baseline_scores, treatment_scores):
            if rng.random() < 0.5:
                left.append(base)
                right.append(treatment)
            else:
                left.append(treatment)
                right.append(base)
        if abs(_mean_delta(left, right)) >= observed:
            extreme += 1
    return {
        "mean_delta": _mean_delta(baseline_scores, treatment_scores),
        "n": len(baseline_scores),
        "p_value": (extreme + 1.0) / float(num_rounds + 1.0),
        "test": "paired_randomization",
    }


def paired_bootstrap_ci(
    baseline_scores: list[float],
    treatment_scores: list[float],
    *,
    num_rounds: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Bootstrap confidence interval for paired mean deltas."""

    if len(baseline_scores) != len(treatment_scores):
        raise ValueError("baseline_scores and treatment_scores must have the same length")
    n = len(baseline_scores)
    if n < 2:
        return {"ci_low": None, "ci_high": None, "warning": "insufficient_sample", "n": n}
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(int(num_rounds)):
        indices = [rng.randrange(n) for _ in range(n)]
        left = [baseline_scores[index] for index in indices]
        right = [treatment_scores[index] for index in indices]
        deltas.append(_mean_delta(left, right))
    deltas.sort()
    low = deltas[int((alpha / 2.0) * len(deltas))]
    high = deltas[min(len(deltas) - 1, int((1.0 - alpha / 2.0) * len(deltas)))]
    return {"ci_high": high, "ci_low": low, "mean_delta": _mean_delta(baseline_scores, treatment_scores), "n": n}


def _mean_delta(baseline_scores: list[float], treatment_scores: list[float]) -> float:
    return sum(float(t) - float(b) for b, t in zip(baseline_scores, treatment_scores)) / float(len(baseline_scores) or 1)
