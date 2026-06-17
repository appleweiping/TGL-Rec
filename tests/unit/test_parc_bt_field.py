"""Unit tests for PaRC Bradley-Terry field + symmetrization (CPU, deterministic)."""

from __future__ import annotations

import numpy as np

from llm4rec.methods.parc.bt_field import fit_bt_field
from llm4rec.methods.parc.config import PaRCConfig
from llm4rec.methods.parc.pairwise_prompt import symmetrized_duel


def _full_pairs_from_theta(theta: np.ndarray, noise: float = 0.0, seed: int = 0):
    """All-pairs symmetrized observations s_ij = theta_i - theta_j (+ noise)."""
    rng = np.random.default_rng(seed)
    pairs = []
    K = len(theta)
    for i in range(K):
        for j in range(i + 1, K):
            s = float(theta[i] - theta[j])
            if noise:
                s += float(rng.normal(0.0, noise))
            pairs.append((i, j, s))
    return pairs


def test_bt_recovers_known_theta_transitive():
    """BT MLE recovers a known transitive theta within tolerance (no anchor)."""
    rng = np.random.default_rng(42)
    K = 12
    theta_true = rng.normal(size=K)
    theta_true = theta_true - theta_true.mean()  # gauge: mean-centred
    pairs = _full_pairs_from_theta(theta_true, noise=0.0)

    res = fit_bt_field(pairs, np.zeros(K), n_items=K, l2_beta=1e-4, fit_alpha=False, anchor=False)
    # recovered field (beta == theta in the no-anchor mode), mean-centred
    recovered = res.beta - res.beta.mean()
    assert np.allclose(recovered, theta_true, atol=1e-2), (
        f"max err {np.max(np.abs(recovered - theta_true)):.4f}"
    )


def test_bt_recovers_theta_with_small_noise():
    """Order is recovered exactly under modest comparison noise."""
    rng = np.random.default_rng(7)
    K = 15
    theta_true = np.sort(rng.normal(size=K))[::-1]  # clearly separated
    theta_true = theta_true - theta_true.mean()
    pairs = _full_pairs_from_theta(theta_true, noise=0.05, seed=7)
    res = fit_bt_field(pairs, np.zeros(K), n_items=K, l2_beta=1e-3, fit_alpha=False, anchor=False)
    rec_order = np.argsort(-res.beta)
    true_order = np.argsort(-theta_true)
    assert list(rec_order) == list(true_order)


def test_bt_anchor_decomposition_recovers_alpha():
    """theta = alpha*pony + beta: when truth is alpha*pony, beta ~ 0 and alpha recovered."""
    rng = np.random.default_rng(3)
    K = 20
    pony = rng.normal(size=K)
    alpha_true = 1.7
    theta_true = alpha_true * pony  # pure anchor, no residual
    pairs = _full_pairs_from_theta(theta_true, noise=0.0)
    res = fit_bt_field(pairs, pony, n_items=K, l2_beta=1.0, fit_alpha=True, anchor=True)
    assert abs(res.alpha - alpha_true) < 0.1, f"alpha={res.alpha}"
    # beta is the residual over the anchor -> small relative to theta
    assert res.order_var_fraction_beta < 0.1


def test_bt_beta_captures_residual_over_anchor():
    """A genuine comparative residual orthogonal to pony shows up in beta."""
    rng = np.random.default_rng(11)
    K = 20
    pony = rng.normal(size=K)
    beta_true = rng.normal(size=K)
    beta_true = beta_true - beta_true.mean()
    theta_true = 1.0 * pony + beta_true
    pairs = _full_pairs_from_theta(theta_true, noise=0.0)
    res = fit_bt_field(pairs, pony, n_items=K, l2_beta=1e-2, fit_alpha=True, anchor=True)
    # recovered beta correlates with the true residual
    corr = np.corrcoef(res.beta, beta_true)[0, 1]
    assert corr > 0.9, f"beta corr {corr:.3f}"
    assert res.var_beta > 0.0


def test_bt_handles_sparse_partial_graph():
    """A sparse/partial duel graph (chain only) still fits without error and shrinks."""
    K = 10
    theta_true = np.linspace(2, -2, K)
    # only adjacent comparisons observed (a connected chain, no all-pairs)
    pairs = [(i, i + 1, float(theta_true[i] - theta_true[i + 1])) for i in range(K - 1)]
    res = fit_bt_field(pairs, np.zeros(K), n_items=K, l2_beta=0.1, fit_alpha=False, anchor=False)
    # chain is order-informative: monotone order preserved
    assert list(np.argsort(-res.beta)) == list(range(K))


def test_bt_no_duels_recovers_pony():
    """Empty duel graph -> beta=0, theta = pony (floor)."""
    pony = np.array([3.0, 1.0, 2.0, 0.5])
    res = fit_bt_field([], pony, n_items=4, l2_beta=1.0, fit_alpha=True, anchor=True)
    assert np.allclose(res.beta, 0.0)
    assert np.allclose(res.theta, pony)


# --------------------------------------------------------------------------- #
# Symmetrization cancels a known position bias.
# --------------------------------------------------------------------------- #
class _BiasedDuelModel:
    """duel_logit = true_pref(A,B) + constant slot-A bias.

    The genuine i-over-j preference is ``pref[i] - pref[j]`` (antisymmetric under
    swap); a constant additive ``slot_a_bias`` is added because the item is in
    slot A (does NOT flip under swap). Symmetrization must cancel the bias.
    """

    def __init__(self, pref: dict[str, float], slot_a_bias: float):
        self.pref = pref
        self.slot_a_bias = slot_a_bias

    def duel_logit(self, prompt: str) -> float:
        body = prompt.split("Two candidate items:", 1)[-1]
        a = body.split("A:", 1)[-1].split("B:", 1)[0].strip()
        b = body.split("B:", 1)[-1].split("Given", 1)[0].strip()
        # strip the truncation ellipsis if present
        a = a.rstrip("…").strip()
        b = b.rstrip("…").strip()
        return (self.pref.get(a, 0.0) - self.pref.get(b, 0.0)) + self.slot_a_bias


def test_symmetrization_cancels_position_bias():
    cfg = PaRCConfig()
    pref = {"itemX": 2.0, "itemY": -1.0}
    bias = 5.0  # huge constant slot-A pull
    model = _BiasedDuelModel(pref, slot_a_bias=bias)
    out = symmetrized_duel(["h1", "h2"], "itemX", "itemY", model, cfg)
    # genuine preference is pref[X]-pref[Y] = 3.0; bias must be removed.
    assert abs(out["s_ij"] - 3.0) < 1e-9, out
    # the diagnostic recovers the bias magnitude.
    assert abs(out["position_bias"] - bias) < 1e-9, out


def test_one_directional_keeps_position_bias():
    """With symmetrize OFF, the bias contaminates s_ij (the ablation cost)."""
    cfg = PaRCConfig(symmetrize=False)
    model = _BiasedDuelModel({"itemX": 2.0, "itemY": -1.0}, slot_a_bias=5.0)
    out = symmetrized_duel(["h1"], "itemX", "itemY", model, cfg)
    # one-directional logit = 3.0 + 5.0 bias = 8.0 (bias not cancelled)
    assert abs(out["s_ij"] - 8.0) < 1e-9, out
