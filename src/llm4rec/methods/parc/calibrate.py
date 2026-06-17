"""Validation-gated lambda-mixture calibration + intransitivity diagnostics for PaRC.

Two responsibilities:

1. ``select_lambda`` / ``mixture_score`` -- the EMPIRICAL FLOOR.
   Final score ``score_i = pony_i + lambda * beta_i``, with ``lambda >= 0``
   chosen on a VALIDATION split to maximise NDCG@k. ``lambda = 0`` is in the grid
   and exactly recovers the pony order (floor = pony). beta may be standardized
   (z-scored) first so lambda is on a pony-comparable scale.

2. ``cyclic_triple_rate`` + ``intransitivity_test`` -- the SCIENTIFIC diagnostic.
   A frozen LLM that just subtracts two pointwise scores is BT-transitive; genuine
   CYCLIC intransitivity (i>j>k>i) is order signal NO pointwise utility can
   reproduce. We measure the cyclic-triple rate on the symmetrized duel signs and
   test it against a STRICT null = a fitted BT field + per-pair heteroskedastic
   comparison noise (NOT a generic equal-noise null), with a bootstrap CI on the
   cyclicity. Significant excess cyclicity over this null => the pairwise field is
   non-BT.

Pure numpy + the project's existing metric helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from llm4rec.methods.parc.bt_field import fit_bt_field


# --------------------------------------------------------------------------- #
# Ranking metric (NDCG@k) on a same-candidate panel.
# --------------------------------------------------------------------------- #
def ndcg_at_k(scores: np.ndarray, pos_index: int, k: int) -> float:
    """Single-positive NDCG@k on one panel (binary relevance, one relevant item).

    Rank by score desc (deterministic index tie-break); gain 1/log2(rank+1) if the
    positive lands in the top-k, else 0. Matches the project's same-candidate
    101-protocol metric (1 positive among K).
    """
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    rank = order.index(pos_index) + 1  # 1-based
    return float(1.0 / np.log2(rank + 1)) if rank <= k else 0.0


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-12 else x - x.mean()


def mixture_score(
    pony: np.ndarray,
    beta: np.ndarray,
    lam: float,
    *,
    standardize_beta: bool = True,
) -> np.ndarray:
    """``score_i = pony_i + lam * beta_i`` (beta optionally z-scored to pony scale)."""
    pony = np.asarray(pony, dtype=float)
    b = _standardize(beta) if standardize_beta else np.asarray(beta, dtype=float)
    return pony + lam * b


@dataclass
class LambdaSelection:
    """Result of validation lambda selection."""

    lam: float
    val_metric: float
    pony_metric: float
    lift: float                 # val_metric - pony_metric
    curve: list[tuple[float, float]]   # (lambda, metric) sweep
    collapsed: bool             # True if best lambda is ~0 (KILL signal)


def select_lambda(
    val_panels: list[dict],
    *,
    lambda_grid: tuple[float, ...],
    k: int = 10,
    standardize_beta: bool = True,
    collapse_tol: float = 1e-9,
) -> LambdaSelection:
    """Pick lambda>=0 maximising mean NDCG@k over validation panels.

    Each panel dict: ``{"pony": np.ndarray[K], "beta": np.ndarray[K],
    "pos_index": int}``. lambda=0 (must be in grid) recovers the pony order.
    Reports whether the selected lambda collapsed to ~0 (the plan's KILL signal).
    """
    grid = sorted(set(lambda_grid))
    if 0.0 not in grid:
        grid = [0.0] + grid

    def mean_metric(lam: float) -> float:
        vals = [
            ndcg_at_k(
                mixture_score(p["pony"], p["beta"], lam, standardize_beta=standardize_beta),
                int(p["pos_index"]),
                k,
            )
            for p in val_panels
        ]
        return float(np.mean(vals)) if vals else 0.0

    curve = [(lam, mean_metric(lam)) for lam in grid]
    pony_metric = next(m for lam, m in curve if lam == 0.0)
    # argmax with a tie-break toward the SMALLEST lambda (parsimony / floor)
    best_lam, best_m = max(curve, key=lambda lm: (lm[1], -lm[0]))
    return LambdaSelection(
        lam=best_lam,
        val_metric=best_m,
        pony_metric=pony_metric,
        lift=best_m - pony_metric,
        curve=curve,
        collapsed=(best_lam <= collapse_tol),
    )


# --------------------------------------------------------------------------- #
# Intransitivity diagnostic.
# --------------------------------------------------------------------------- #
def cyclic_triple_rate(
    pairs: list[tuple[int, int, float]],
    n_items: int,
    *,
    sample: int = 0,
    rng: np.random.Generator | None = None,
) -> float:
    """Fraction of fully-determined triples (i,j,k) that form a preference CYCLE.

    Builds a sign matrix from symmetrized ``s_ij`` (sign>0 => i beats j). A triple
    is cyclic iff i>j, j>k, k>i (or the reverse cycle). Only triples whose three
    edges are all observed and non-tied are counted. ``sample>0`` randomly samples
    that many triples (for large K); 0 = exhaustive.
    """
    sign = np.zeros((n_items, n_items))
    observed = np.zeros((n_items, n_items), dtype=bool)
    for (i, j, s) in pairs:
        if s == 0.0:
            continue
        sign[i, j] = np.sign(s)
        sign[j, i] = -np.sign(s)
        observed[i, j] = observed[j, i] = True

    def is_cyclic(a: int, b: int, c: int) -> bool | None:
        if not (observed[a, b] and observed[b, c] and observed[a, c]):
            return None
        # cyclic iff a>b>c>a  OR  a<b<c<a   <=>  sign(ab)==sign(bc)==sign(ca)
        return sign[a, b] == sign[b, c] == sign[c, a]

    n_tri = 0
    n_cyc = 0
    if sample and sample > 0:
        rng = rng or np.random.default_rng(0)
        tries = 0
        max_tries = sample * 20
        while n_tri < sample and tries < max_tries:
            tries += 1
            a, b, c = sorted(rng.choice(n_items, size=3, replace=False).tolist())
            res = is_cyclic(a, b, c)
            if res is None:
                continue
            n_tri += 1
            n_cyc += int(res)
    else:
        for a in range(n_items):
            for b in range(a + 1, n_items):
                for c in range(b + 1, n_items):
                    res = is_cyclic(a, b, c)
                    if res is None:
                        continue
                    n_tri += 1
                    n_cyc += int(res)
    return float(n_cyc) / n_tri if n_tri else 0.0


@dataclass
class IntransitivityResult:
    """Observed cyclic rate vs the BT+heteroskedastic-noise null."""

    observed_rate: float
    null_mean: float
    null_ci_low: float
    null_ci_high: float
    p_value: float              # P(null_rate >= observed) -- one-sided excess test
    significant: bool           # observed above the null 95% upper CI


def intransitivity_test(
    pairs: list[tuple[int, int, float]],
    n_items: int,
    *,
    pony: np.ndarray | None = None,
    n_bootstrap: int = 500,
    seed: int = 20260506,
    l2_beta: float = 1.0,
    triple_sample: int = 0,
) -> IntransitivityResult:
    """Test observed cyclic-triple rate vs a strict BT + heteroskedastic-noise null.

    Null construction (per the plan -- NOT a generic equal-noise BT null):
      1. Fit a BT field theta to the observed symmetrized ``s_ij`` (this is the
         best transitive explanation of the data).
      2. Estimate a PER-PAIR heteroskedastic noise scale from the BT residuals,
         bucketed by the fitted utility gap |theta_i - theta_j| (close pairs are
         noisier -- the realistic model of LLM comparison noise).
      3. Resample synthetic ``s_ij ~ (theta_i - theta_j) + N(0, sigma(gap))`` on the
         SAME duel graph, recompute the cyclic rate. Repeat ``n_bootstrap`` times.
    A purely-BT generator still produces SOME cycles from noise; significance =
    the observed rate exceeds the 95th percentile of this strict null.
    """
    rng = np.random.default_rng(seed)
    observed = cyclic_triple_rate(pairs, n_items, sample=triple_sample, rng=rng)
    if not pairs:
        return IntransitivityResult(observed, 0.0, 0.0, 0.0, 1.0, False)

    anchor = pony is not None
    pony_vec = np.asarray(pony, dtype=float) if anchor else np.zeros(n_items)
    fit = fit_bt_field(
        pairs, pony_vec, n_items=n_items, l2_beta=l2_beta,
        fit_alpha=anchor, anchor=anchor,
    )
    theta = fit.theta

    # heteroskedastic residual scale bucketed by |theta gap|
    gaps = np.array([abs(theta[i] - theta[j]) for (i, j, _) in pairs])
    resid = np.array([s - (theta[i] - theta[j]) for (i, j, s) in pairs])
    # bucket gaps into terciles; sigma per bucket (close pairs -> larger sigma).
    if len(gaps) >= 3 and np.ptp(gaps) > 1e-12:
        edges = np.quantile(gaps, [1 / 3, 2 / 3])
        bucket = np.digitize(gaps, edges)
    else:
        bucket = np.zeros(len(gaps), dtype=int)
    sigma_by_bucket: dict[int, float] = {}
    for bkt in np.unique(bucket):
        rs = resid[bucket == bkt]
        sigma_by_bucket[int(bkt)] = float(rs.std()) if len(rs) > 1 else float(resid.std() + 1e-9)

    null_rates = np.empty(n_bootstrap)
    pair_idx = list(range(len(pairs)))
    for b in range(n_bootstrap):
        synth: list[tuple[int, int, float]] = []
        for pi in pair_idx:
            i, j, _ = pairs[pi]
            sig = sigma_by_bucket.get(int(bucket[pi]), float(resid.std() + 1e-9))
            s_synth = (theta[i] - theta[j]) + rng.normal(0.0, max(sig, 1e-9))
            synth.append((i, j, float(s_synth)))
        null_rates[b] = cyclic_triple_rate(synth, n_items, sample=triple_sample, rng=rng)

    null_mean = float(null_rates.mean())
    ci_low = float(np.percentile(null_rates, 2.5))
    ci_high = float(np.percentile(null_rates, 97.5))
    p_value = float((np.sum(null_rates >= observed) + 1) / (n_bootstrap + 1))
    return IntransitivityResult(
        observed_rate=observed,
        null_mean=null_mean,
        null_ci_low=ci_low,
        null_ci_high=ci_high,
        p_value=p_value,
        significant=bool(observed > ci_high),
    )
