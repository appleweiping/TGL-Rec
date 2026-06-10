"""Conformal calibration / abstention layer for CC-PACE.

HONEST framing (docs/method_v2_decision_CC-PACE.md): on a closed 101-panel, the
within-panel NDCG depends only on the ORDER of the ranked statistic T. A conformal
p-value is a monotone transform of T within the panel, so it does NOT change NDCG.
Conformal here is a SELECTIVE-PREDICTION / coverage layer, not the ranking engine.
Ranking is driven by the residualized statistic T (statistic.py).

What this module provides:
  - per-candidate conformal p-value against the panel empirical null (P_pop);
  - optional dual null merge with an auxiliary reference panel (P_sem) via
    intersection-union (max p) or e-value averaging;
  - James-Stein shrinkage for sparse users.
"""

from __future__ import annotations

import numpy as np


def conformal_p(stat: np.ndarray, idx: int) -> float:
    """Conformal p-value for candidate ``idx`` vs the panel null (all candidates).

    p = (1 + #{j: stat_j >= stat_idx}) / (n + 1) over j != idx, super-uniform
    under H0-exchangeability. Smaller p = more anomalous = better.
    """
    n = len(stat)
    others = np.delete(stat, idx)
    ge = int(np.sum(others >= stat[idx]))
    return (1.0 + ge) / (n + 1.0)


def all_conformal_p(stat: np.ndarray) -> np.ndarray:
    return np.array([conformal_p(stat, i) for i in range(len(stat))])


def merge_pvalues(p_pop: np.ndarray, p_sem: np.ndarray, *, mode: str = "max") -> np.ndarray:
    """Combine two panels' p-values conservatively (no independence assumed).

    - "max": intersection-union test, P(max p <= a) <= a for either null.
    - "evalue_avg": E_k = 1/p_k capped, average, p = min(1, 1/E_avg) (Ville/Markov).
    """
    if mode == "max":
        return np.maximum(p_pop, p_sem)
    if mode == "evalue_avg":
        e_pop = np.minimum(1.0 / np.clip(p_pop, 1e-6, 1.0), 1e6)
        e_sem = np.minimum(1.0 / np.clip(p_sem, 1e-6, 1.0), 1e6)
        e_avg = 0.5 * (e_pop + e_sem)
        return np.minimum(1.0, 1.0 / np.clip(e_avg, 1e-9, None))
    raise ValueError(f"unknown merge mode: {mode}")


def james_stein_shrink(
    stat: np.ndarray,
    uncertainty: np.ndarray,
    facet_bucket: np.ndarray | None,
    *,
    n_history: int,
    n_cap: int = 20,
) -> np.ndarray:
    """Empirical-Bayes James-Stein shrinkage toward the facet-bucket mean.

    Shrinks more for sparse users (small ``n_history``) and high-uncertainty
    candidates. Rich users -> no shrink; cold-start -> shrink to bucket mean.
    """
    n = len(stat)
    if facet_bucket is not None and len(facet_bucket) == n:
        target = np.empty(n)
        for b in np.unique(facet_bucket):
            mask = facet_bucket == b
            target[mask] = stat[mask].mean()
    else:
        target = np.full(n, float(stat.mean()))

    hist_factor = min(max(n_history, 0), n_cap) / float(n_cap)  # 0 (cold) .. 1 (rich)
    var = np.clip(uncertainty, 1e-6, None) ** 2
    var_norm = var / (var.mean() + 1e-9)
    b_u = (1.0 - hist_factor) * (var_norm / (var_norm + 1.0))
    b_u = np.clip(b_u, 0.0, 1.0)
    return (1.0 - b_u) * stat + b_u * target
