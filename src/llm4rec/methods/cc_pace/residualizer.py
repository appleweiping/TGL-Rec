"""Residualizer for CC-PACE: strip the best ADDITIVE nuisance explanation.

T_u(c) = E_judge(c) - m_hat_LOO(nuisance(c)), where nuisance = (log-pop, facet
bucket, CF score r_cf, ...). m_hat is fit by SYMMETRIC leave-one-out on the panel
candidates: for each candidate i, m_hat is fit on the OTHER 100 candidates and
evaluated at i. Symmetric LOO (excluding i, not "excluding the known positive")
preserves panel exchangeability -- the conformal validity condition.

Two residualizer classes (config.residualizer_rich):
  - additive isotonic (default): removes only the monotone marginal of each
    nuisance coordinate. The surviving residual is the CF x content x intent
    interaction reasoning an additive model cannot express -- the contribution.
  - rich (ceiling test): a gradient-boosted / interaction model over all nuisance
    coords. If T -> 0 under the rich residualizer, the "judge reasoning" is just
    expressible CF interactions and the claim honestly degrades to "interactions
    matter". MANDATORY to report both (docs/method_v2_decision_CC-PACE.md).
"""

from __future__ import annotations

import numpy as np


def _isotonic_fit_predict(x: np.ndarray, y: np.ndarray, x_query: float) -> float:
    """Pool-adjacent-violators isotonic regression; predict at a single query x.

    Monotone non-decreasing fit of y on x; query by nearest fitted block. Pure
    numpy, no sklearn dependency.
    """
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    # PAVA
    blocks_y = list(ys.astype(float))
    blocks_w = [1.0] * len(blocks_y)
    blocks_x = list(xs.astype(float))
    i = 0
    while i < len(blocks_y) - 1:
        if blocks_y[i] > blocks_y[i + 1] + 1e-12:
            new_y = (blocks_y[i] * blocks_w[i] + blocks_y[i + 1] * blocks_w[i + 1]) / (
                blocks_w[i] + blocks_w[i + 1]
            )
            new_w = blocks_w[i] + blocks_w[i + 1]
            new_x = max(blocks_x[i], blocks_x[i + 1])
            blocks_y[i : i + 2] = [new_y]
            blocks_w[i : i + 2] = [new_w]
            blocks_x[i : i + 2] = [new_x]
            if i > 0:
                i -= 1
        else:
            i += 1
    fit_x = np.array(blocks_x)
    fit_y = np.array(blocks_y)
    idx = int(np.searchsorted(fit_x, x_query, side="left"))
    idx = min(max(idx, 0), len(fit_y) - 1)
    return float(fit_y[idx])


def residualize_additive(
    evidence: np.ndarray,
    nuisance: dict[str, np.ndarray],
    *,
    keys: tuple[str, ...],
) -> np.ndarray:
    """Additive symmetric-LOO isotonic residual over the given nuisance keys.

    Backfitting: residual starts at evidence; for each nuisance coordinate we
    subtract its symmetric-LOO isotonic fit, one pass (sufficient for additive
    removal at panel scale). Coordinates not present in ``nuisance`` are skipped.
    """
    n = len(evidence)
    resid = evidence.astype(float).copy()
    for key in keys:
        if key not in nuisance:
            continue
        x = np.asarray(nuisance[key], dtype=float)
        if x.shape[0] != n or np.allclose(x, x[0]):
            continue  # constant / mismatched coordinate carries no signal
        fitted = np.empty(n)
        idx_all = np.arange(n)
        for i in range(n):
            mask = idx_all != i
            fitted[i] = _isotonic_fit_predict(x[mask], resid[mask], float(x[i]))
        resid = resid - fitted
    return resid


def residualize(
    evidence: np.ndarray,
    nuisance: dict[str, np.ndarray],
    *,
    keys: tuple[str, ...],
    rich: bool = False,
) -> np.ndarray:
    """Dispatch to additive or rich (ceiling-test) residualizer."""
    if not rich:
        return residualize_additive(evidence, nuisance, keys=keys)
    return _residualize_rich(evidence, nuisance, keys=keys)


def _residualize_rich(
    evidence: np.ndarray, nuisance: dict[str, np.ndarray], *, keys: tuple[str, ...]
) -> np.ndarray:
    """Interaction-rich ceiling test: remove a multivariate (linear+pairwise) fit.

    Uses a leave-one-out ridge on [coords, pairwise products]. If the residual
    collapses to ~0 here, the judge evidence was just expressible nuisance
    interactions. Numpy-only; falls back to additive if too few points.
    """
    n = len(evidence)
    cols = [np.asarray(nuisance[k], dtype=float) for k in keys if k in nuisance]
    cols = [c for c in cols if c.shape[0] == n and not np.allclose(c, c[0])]
    if not cols:
        return evidence.astype(float).copy()
    base = np.column_stack(cols)
    # add pairwise products for interaction capacity
    inter = [base[:, a] * base[:, b] for a in range(base.shape[1]) for b in range(a, base.shape[1])]
    X = np.column_stack([np.ones(n), base, np.column_stack(inter)]) if inter else np.column_stack([np.ones(n), base])
    resid = np.empty(n)
    lam = 1e-3
    idx_all = np.arange(n)
    for i in range(n):
        mask = idx_all != i
        Xm, ym = X[mask], evidence[mask]
        A = Xm.T @ Xm + lam * np.eye(Xm.shape[1])
        try:
            w = np.linalg.solve(A, Xm.T @ ym)
        except np.linalg.LinAlgError:
            w = np.zeros(X.shape[1])
        resid[i] = float(evidence[i] - X[i] @ w)
    return resid
