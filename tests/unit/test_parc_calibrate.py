"""Unit tests for PaRC lambda-mixture calibration + intransitivity (CPU, deterministic)."""

from __future__ import annotations

import numpy as np

from llm4rec.methods.parc.calibrate import (
    cyclic_triple_rate,
    intransitivity_test,
    mixture_score,
    ndcg_at_k,
    select_lambda,
)


# --------------------------------------------------------------------------- #
# lambda-mixture floor behaviour.
# --------------------------------------------------------------------------- #
def test_lambda_zero_recovers_pony_order():
    rng = np.random.default_rng(0)
    K = 101
    pony = rng.normal(size=K)
    beta = rng.normal(size=K)  # arbitrary residual
    s0 = mixture_score(pony, beta, 0.0, standardize_beta=True)
    # lambda=0 -> score == pony exactly -> identical order
    assert np.allclose(s0, pony)
    assert list(np.argsort(-s0)) == list(np.argsort(-pony))


def test_lambda_positive_shifts_toward_beta():
    K = 10
    pony = np.zeros(K)               # anchor totally flat
    beta = np.arange(K, dtype=float)  # beta fully determines order when lam>0
    s_lam = mixture_score(pony, beta, 1.0, standardize_beta=False)
    # with a flat anchor, a positive lambda makes the order follow beta
    assert list(np.argsort(-s_lam)) == list(np.argsort(-beta))


def test_select_lambda_picks_zero_when_beta_is_noise():
    """If beta carries no validation signal, lambda collapses to 0 (KILL signal)."""
    rng = np.random.default_rng(1)
    panels = []
    for _ in range(80):
        K = 20
        pony = rng.normal(size=K)
        pos = int(np.argmax(pony))         # positive aligns with pony (pony already good)
        beta = rng.normal(size=K)          # pure noise, uncorrelated with the label
        panels.append({"pony": pony, "beta": beta, "pos_index": pos})
    sel = select_lambda(panels, lambda_grid=(0.0, 0.1, 0.5, 1.0, 2.0), k=10)
    assert sel.lam == 0.0
    assert sel.collapsed is True
    assert abs(sel.lift) < 1e-9


def test_select_lambda_picks_positive_when_beta_helps():
    """If beta points at the positive that pony mis-ranks, lambda>0 is selected."""
    rng = np.random.default_rng(2)
    panels = []
    for _ in range(80):
        K = 20
        pony = rng.normal(size=K)
        pos = rng.integers(K)
        # pony ranks the positive poorly; beta points strongly at it.
        pony[pos] = pony.min() - 1.0
        beta = np.zeros(K)
        beta[pos] = 5.0
        panels.append({"pony": pony, "beta": beta, "pos_index": int(pos)})
    sel = select_lambda(panels, lambda_grid=(0.0, 0.1, 0.5, 1.0, 2.0), k=10)
    assert sel.lam > 0.0
    assert sel.collapsed is False
    assert sel.lift > 0.0
    assert sel.val_metric > sel.pony_metric


def test_ndcg_at_k_basic():
    scores = np.array([0.1, 0.9, 0.5, 0.2])  # positive at idx1 ranks #1
    assert ndcg_at_k(scores, 1, 10) == 1.0
    # positive at idx0 (lowest among >0.1?) -> rank 4, outside k=2 -> 0
    assert ndcg_at_k(scores, 0, 2) == 0.0


# --------------------------------------------------------------------------- #
# Intransitivity metric.
# --------------------------------------------------------------------------- #
def _transitive_pairs(theta):
    K = len(theta)
    return [(i, j, float(theta[i] - theta[j])) for i in range(K) for j in range(i + 1, K)]


def test_cyclic_rate_zero_on_transitive_data():
    theta = np.linspace(5, -5, 12)
    pairs = _transitive_pairs(theta)
    rate = cyclic_triple_rate(pairs, len(theta))
    assert rate == 0.0


def test_cyclic_rate_detects_injected_cycle():
    """A planted 3-cycle on otherwise-transitive data yields a nonzero rate."""
    theta = np.linspace(5, -5, 8)
    pairs = dict(((i, j), float(theta[i] - theta[j])) for i in range(8) for j in range(i + 1, 8))
    # inject a cycle among {0,1,2}: force 2>0 (was 0>2), keeping 0>1, 1>2 -> 0>1>2>0
    pairs[(0, 2)] = -abs(theta[0] - theta[2]) - 1.0  # s_02 < 0 => 2 beats 0
    pair_list = [(i, j, s) for (i, j), s in pairs.items()]
    rate = cyclic_triple_rate(pair_list, 8)
    assert rate > 0.0


def test_intransitivity_significant_for_cyclic_field_vs_transitive_control():
    """The test flags a genuinely cyclic field and clears a transitive control."""
    rng = np.random.default_rng(20260506)
    K = 9

    # --- transitive control: s_ij = theta_i - theta_j + small noise ---
    theta = rng.normal(size=K) * 2.0
    ctrl_pairs = [
        (i, j, float(theta[i] - theta[j] + rng.normal(0, 0.05)))
        for i in range(K)
        for j in range(i + 1, K)
    ]
    ctrl = intransitivity_test(ctrl_pairs, K, pony=None, n_bootstrap=300, triple_sample=0)
    assert ctrl.significant is False, (
        f"transitive control wrongly flagged: obs={ctrl.observed_rate:.3f} "
        f"ci_high={ctrl.null_ci_high:.3f}"
    )

    # --- cyclic field: a rotational tournament (strongly non-BT) ---
    # Item i beats the next floor(K/2) items cyclically, with CONSISTENT unit
    # margins and small noise. No transitive utility reproduces this order, and
    # because the margins are consistent (not noise-driven), independent-noise
    # resampling on the same graph rarely reconstructs the same cycles -> the
    # observed cyclic rate is a clear outlier above the strict BT+noise null.
    half = K // 2
    cyc_pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            fwd = (j - i) % K
            s = 1.0 if fwd <= half else -1.0
            s += rng.normal(0, 0.02)
            cyc_pairs.append((i, j, float(s)))
    cyc = intransitivity_test(cyc_pairs, K, pony=None, n_bootstrap=400, triple_sample=0)
    assert cyc.observed_rate > ctrl.observed_rate
    assert cyc.significant is True, (
        f"cyclic field not flagged: obs={cyc.observed_rate:.3f} "
        f"ci_high={cyc.null_ci_high:.3f} p={cyc.p_value:.3f}"
    )
