"""Significance-test interfaces for paired recommendation metrics."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
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


@dataclass
class PairedMoments:
    """Streaming moments for paired metric differences."""

    n: int = 0
    sum_delta: float = 0.0
    sum_delta_sq: float = 0.0

    def add(self, baseline_score: float, treatment_score: float) -> None:
        delta = float(treatment_score) - float(baseline_score)
        self.n += 1
        self.sum_delta += delta
        self.sum_delta_sq += delta * delta

    def extend(self, baseline_scores: list[float], treatment_scores: list[float]) -> None:
        if len(baseline_scores) != len(treatment_scores):
            raise ValueError("baseline_scores and treatment_scores must have the same length")
        for baseline, treatment in zip(baseline_scores, treatment_scores):
            self.add(baseline, treatment)


def paired_t_test(
    baseline_scores: list[float],
    treatment_scores: list[float],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Two-sided paired t-test over per-example metric contributions."""

    moments = PairedMoments()
    moments.extend(baseline_scores, treatment_scores)
    return paired_t_test_from_moments(moments, alpha=alpha)


def paired_t_test_from_moments(
    moments: PairedMoments,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Return a paired t-test result from streaming delta moments."""

    if moments.n < 2:
        return {
            "effect_direction": "insufficient_sample",
            "mean_delta": None,
            "n": moments.n,
            "notes": "insufficient_sample",
            "p_value": None,
            "significant_at_0_05": False,
            "test": "paired_t_test",
            "warning": "insufficient_sample",
        }
    mean_delta = moments.sum_delta / float(moments.n)
    numerator = moments.sum_delta_sq - float(moments.n) * mean_delta * mean_delta
    variance = max(0.0, numerator / float(max(moments.n - 1, 1)))
    if variance == 0.0:
        p_value = 1.0 if mean_delta == 0.0 else 0.0
        t_statistic = math.inf if mean_delta > 0 else (-math.inf if mean_delta < 0 else 0.0)
        notes = "zero_variance_differences"
    else:
        standard_error = math.sqrt(variance / float(moments.n))
        t_statistic = mean_delta / standard_error
        p_value, notes = _two_sided_t_p_value(t_statistic, moments.n - 1)
    if mean_delta > 0:
        effect_direction = "method_a_better"
    elif mean_delta < 0:
        effect_direction = "method_b_better"
    else:
        effect_direction = "tie"
    return {
        "effect_direction": effect_direction,
        "mean_delta": mean_delta,
        "n": moments.n,
        "notes": notes,
        "p_value": p_value,
        "significant_at_0_05": bool(p_value is not None and p_value < alpha),
        "t_statistic": t_statistic,
        "test": "paired_t_test",
    }


def _two_sided_t_p_value(t_statistic: float, degrees_of_freedom: int) -> tuple[float, str]:
    try:
        from scipy import stats  # type: ignore

        return float(2.0 * stats.t.sf(abs(float(t_statistic)), int(degrees_of_freedom))), "scipy_t_distribution"
    except Exception:
        return math.erfc(abs(float(t_statistic)) / math.sqrt(2.0)), "normal_approximation"


def _mean_delta(baseline_scores: list[float], treatment_scores: list[float]) -> float:
    return sum(float(t) - float(b) for b, t in zip(baseline_scores, treatment_scores)) / float(len(baseline_scores) or 1)
