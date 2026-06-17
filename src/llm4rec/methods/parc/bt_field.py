"""Anchored symmetrized Bradley-Terry field for PaRC.

We observe a set of symmetrized pairwise comparisons ``s_ij`` (i preferred over j;
``s_ij = -s_ji``) over K candidates and model them with a Gaussian / least-squares
Bradley-Terry (Thurstone-style) field:

    s_ij ≈ theta_i - theta_j + noise

decomposed against the pony pointwise posterior as the anchor:

    theta_i = alpha * pony_i + beta_i.

The HEADLINE object is ``beta`` -- the comparative correction the LLM makes only
when forced to compare, the part of the order NOT expressible by the pointwise
posterior. We fit (alpha, beta) by ridge-regularized least squares on the duel
graph (handles sparse / partial graphs; degenerate / disconnected nodes shrink to
0 under the L2 prior). Closed-form normal equations -- no SGD, deterministic.

Diagnostics returned:
  - ``var_beta``: Var(beta) across candidates.
  - ``order_var_fraction_beta``: fraction of the fitted-order variance attributable
    to beta, i.e. Var(beta) / Var(alpha*pony + beta) -- "how much of the comparative
    order is NOT the anchor".
  - ``alpha``: fitted anchor coefficient.
  - ``n_comparisons``, ``residual_rms``.

This is the least-squares (Gaussian-link) BT estimator. It is convex, has a
closed form, recovers true theta on transitive data, and is the natural fit for
the symmetrized real-valued ``s_ij`` (which are log-odds DIFFERENCES, already on
the theta scale -- not 0/1 win indicators). Pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BTFieldResult:
    """Fitted anchored BT field + diagnostics."""

    theta: np.ndarray          # theta_i = alpha*pony_i + beta_i  (length K)
    beta: np.ndarray           # comparative residual (length K, mean-centred)
    alpha: float               # anchor coefficient on pony
    var_beta: float
    order_var_fraction_beta: float
    n_comparisons: int
    residual_rms: float


def fit_bt_field(
    pairs: list[tuple[int, int, float]],
    pony: np.ndarray,
    *,
    n_items: int | None = None,
    l2_beta: float = 1.0,
    fit_alpha: bool = True,
    anchor: bool = True,
) -> BTFieldResult:
    """Fit ``theta_i = alpha*pony_i + beta_i`` from symmetrized duels.

    Args:
      pairs: list of ``(i, j, s_ij)`` with ``s_ij`` the symmetrized i-over-j
        log-odds. Each unordered pair should appear once (i<j by convention is
        fine; ``s_ji = -s_ij`` is implied and need not be duplicated).
      pony: per-candidate pony pointwise posterior (the anchor), length K.
      n_items: K (defaults to ``len(pony)``).
      l2_beta: ridge weight on beta (regularizes sparse / disconnected graphs).
      fit_alpha: if False, alpha is fixed at 1.0.
      anchor: if False, the anchor term is dropped (theta = beta) -- the
        "BT-only, no pony anchor" ablation.

    Model (least squares):
        minimize  sum_{(i,j)} ( (theta_i - theta_j) - s_ij )^2  +  l2_beta * ||beta||^2
        s.t.      theta = alpha*pony + beta   (if anchor)   ;  sum_i beta_i = 0.

    The mean-centring of beta resolves the global shift gauge freedom of
    difference-only observations; the L2 prior resolves any remaining
    rank-deficiency from a disconnected duel graph.
    """
    K = int(n_items if n_items is not None else len(pony))
    pony = np.asarray(pony, dtype=float).reshape(-1)
    if pony.shape[0] != K:
        raise ValueError(f"pony length {pony.shape[0]} != n_items {K}")

    if not anchor:
        # theta = beta directly; pony enters only through l2 prior centre = 0.
        beta, alpha, rms, ncmp = _fit_difference_field(pairs, K, l2_beta)
        theta = beta.copy()
        return _package(theta, beta, 0.0, K, ncmp, rms, pony, anchored=False)

    if not pairs:
        # no duels -> beta shrinks fully to 0, theta = alpha*pony with alpha=1.
        alpha = 1.0
        beta = np.zeros(K)
        theta = alpha * pony + beta
        return _package(theta, beta, alpha, K, 0, 0.0, pony, anchored=True)

    # Parameterize unknowns as [alpha (optional), beta_0..beta_{K-1}].
    # Each duel (i,j,s) contributes:
    #   (theta_i - theta_j) - s
    #   = alpha*(pony_i - pony_j) + (beta_i - beta_j) - s
    # Plus an L2 ridge on each beta_k, and a soft sum(beta)=0 centring constraint.
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    p_alpha = 1 if fit_alpha else 0
    dim = p_alpha + K

    for (i, j, s) in pairs:
        row = np.zeros(dim)
        if fit_alpha:
            row[0] = pony[i] - pony[j]
        else:
            # alpha fixed at 1.0: fold the known anchor diff into the rhs.
            s = s - (pony[i] - pony[j])
        row[p_alpha + i] += 1.0
        row[p_alpha + j] -= 1.0
        rows.append(row)
        rhs.append(float(s))

    A = np.asarray(rows, dtype=float)
    b = np.asarray(rhs, dtype=float)

    # Normal equations with ridge on beta + centring. A TINY ridge also lands on
    # alpha so the system stays non-singular when the anchor is constant /
    # degenerate (pony_i - pony_j == 0 for all duels => zero alpha column).
    AtA = A.T @ A
    Atb = A.T @ b
    reg = np.zeros((dim, dim))
    if fit_alpha:
        reg[0, 0] = 1e-8
    for k in range(K):
        reg[p_alpha + k, p_alpha + k] = l2_beta
    # soft centring: large weight * (sum beta)^2 to pin the gauge.
    c = np.zeros(dim)
    c[p_alpha:] = 1.0
    centring_w = 1e3
    AtA = AtA + reg + centring_w * np.outer(c, c)

    sol = np.linalg.solve(AtA, Atb)
    alpha = float(sol[0]) if fit_alpha else 1.0
    beta = sol[p_alpha:]
    beta = beta - beta.mean()  # hard re-centre (gauge)

    theta = alpha * pony + beta
    # residual RMS on the observed duels
    pred = np.array([alpha * (pony[i] - pony[j]) + (beta[i] - beta[j]) for (i, j, _) in pairs])
    obs = np.array([s for (_, _, s) in pairs])
    rms = float(np.sqrt(np.mean((pred - obs) ** 2))) if len(obs) else 0.0
    return _package(theta, beta, alpha, K, len(pairs), rms, pony, anchored=True)


def _fit_difference_field(
    pairs: list[tuple[int, int, float]], K: int, l2: float
) -> tuple[np.ndarray, float, float, int]:
    """theta = beta only (no anchor): ridge least squares on difference observations."""
    if not pairs:
        return np.zeros(K), 0.0, 0.0, 0
    rows, rhs = [], []
    for (i, j, s) in pairs:
        row = np.zeros(K)
        row[i] += 1.0
        row[j] -= 1.0
        rows.append(row)
        rhs.append(float(s))
    A = np.asarray(rows, dtype=float)
    b = np.asarray(rhs, dtype=float)
    AtA = A.T @ A + l2 * np.eye(K)
    c = np.ones(K)
    AtA = AtA + 1e3 * np.outer(c, c)
    beta = np.linalg.solve(AtA, A.T @ b)
    beta = beta - beta.mean()
    pred = np.array([beta[i] - beta[j] for (i, j, _) in pairs])
    obs = np.array([s for (_, _, s) in pairs])
    rms = float(np.sqrt(np.mean((pred - obs) ** 2)))
    return beta, 0.0, rms, len(pairs)


def _package(
    theta: np.ndarray,
    beta: np.ndarray,
    alpha: float,
    K: int,
    ncmp: int,
    rms: float,
    pony: np.ndarray,
    *,
    anchored: bool,
) -> BTFieldResult:
    var_beta = float(np.var(beta))
    var_theta = float(np.var(theta))
    # fraction of the fitted-order variance carried by beta (vs the anchor).
    if anchored and var_theta > 1e-12:
        frac = var_beta / var_theta
    elif not anchored:
        frac = 1.0 if var_theta > 1e-12 else 0.0
    else:
        frac = 0.0
    return BTFieldResult(
        theta=theta,
        beta=beta,
        alpha=alpha,
        var_beta=var_beta,
        order_var_fraction_beta=float(min(max(frac, 0.0), 1.0)),
        n_comparisons=ncmp,
        residual_rms=rms,
    )
