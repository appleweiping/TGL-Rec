"""Unit tests for PaRC adaptive duel scheduler (CPU, deterministic)."""

from __future__ import annotations

import numpy as np

from llm4rec.methods.parc.bt_field import fit_bt_field
from llm4rec.methods.parc.config import PaRCConfig
from llm4rec.methods.parc.duels import budget_for, run_duels, schedule_full


def _theta_eval(theta):
    """A noiseless duel evaluator: s_ij = theta_i - theta_j on ORIGINAL indices."""

    def eval_fn(i: int, j: int) -> float:
        return float(theta[i] - theta[j])

    return eval_fn


def test_adaptive_uses_n_log_n_comparisons():
    """Adaptive schedule uses << K^2 comparisons (O(K log K) budget)."""
    K = 101  # the frozen same-candidate protocol size
    cfg = PaRCConfig(duel_mode="adaptive")
    theta = np.linspace(5, -5, K)
    sched = run_duels(theta, _theta_eval(theta), cfg, seed=0)

    full = K * (K - 1) // 2  # 5050
    assert sched.n_comparisons <= budget_for(K, cfg)
    assert sched.n_comparisons < full // 4, (
        f"adaptive used {sched.n_comparisons}, expected << {full}"
    )
    # sanity: budget itself is O(K log K), an order of magnitude below O(K^2)
    assert budget_for(K, cfg) < full


def test_full_mode_is_quadratic():
    K = 12
    theta = np.arange(K, dtype=float)[::-1]
    sched = schedule_full(K, _theta_eval(theta))
    assert sched.n_comparisons == K * (K - 1) // 2
    assert sched.mode == "full"


def test_adaptive_preserves_anchor_order_on_sorted_input():
    """On already-sorted (consistent) input, BT over the adaptive duels keeps the order."""
    K = 60
    cfg = PaRCConfig(duel_mode="adaptive")
    # pony order == true theta order (anchor is already correct)
    theta = np.linspace(10, -10, K)
    pony = theta.copy()
    sched = run_duels(pony, _theta_eval(theta), cfg, seed=1)
    res = fit_bt_field(
        sched.as_pair_list(), pony, n_items=K, l2_beta=cfg.bt_l2_beta,
        fit_alpha=cfg.bt_fit_alpha, anchor=cfg.anchor_pony,
    )
    # theta = alpha*pony + beta should keep the descending order intact
    final_order = np.argsort(-res.theta)
    assert list(final_order) == list(range(K)), "anchor order not preserved"


def test_adaptive_budget_respected_under_small_factor():
    """A tight budget factor strictly caps the comparison count."""
    K = 101
    cfg = PaRCConfig(duel_mode="adaptive", duel_budget_factor=1.5, refine_rounds=5)
    theta = np.random.default_rng(0).normal(size=K)
    sched = run_duels(theta, _theta_eval(theta), cfg, seed=2)
    assert sched.n_comparisons <= budget_for(K, cfg) == sched.budget


def test_adaptive_emits_no_duplicate_pairs():
    K = 40
    cfg = PaRCConfig(duel_mode="adaptive")
    theta = np.random.default_rng(5).normal(size=K)
    sched = run_duels(theta, _theta_eval(theta), cfg, seed=3)
    keys = {(min(i, j), max(i, j)) for (i, j, _) in sched.pairs}
    assert len(keys) == len(sched.pairs), "duplicate pairs emitted"


def test_adaptive_recovers_order_when_pony_anchor_is_wrong():
    """Even with a scrambled anchor, adjacent+boundary duels fix local order near top."""
    K = 30
    cfg = PaRCConfig(duel_mode="adaptive", refine_rounds=3, boundary_window=10)
    theta = np.linspace(5, -5, K)
    rng = np.random.default_rng(9)
    perm = rng.permutation(K)
    pony = np.empty(K)
    pony[perm] = np.linspace(5, -5, K)  # anchor order != theta order
    sched = run_duels(pony, _theta_eval(theta), cfg, seed=4)
    # the duels themselves carry the true signal regardless of anchor; with a
    # non-anchored BT fit over the executed (connected) pairs, top item recovered
    res = fit_bt_field(
        sched.as_pair_list(), pony, n_items=K, l2_beta=1e-2,
        fit_alpha=False, anchor=False,
    )
    # the globally-best item (theta argmax) should be ranked highly by beta
    assert int(np.argmax(res.beta)) == int(np.argmax(theta))
