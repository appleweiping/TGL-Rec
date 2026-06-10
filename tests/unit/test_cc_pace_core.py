"""Unit tests for CC-PACE core components (CPU, no model download)."""

from __future__ import annotations

import numpy as np

from llm4rec.methods.cc_pace import conformal, residualizer
from llm4rec.methods.cc_pace.config import CCPaceConfig


def test_conformal_p_is_super_uniform_under_null():
    rng = np.random.default_rng(0)
    # exchangeable scores -> positive rank uniform -> p roughly uniform
    ps = []
    for _ in range(2000):
        stat = rng.normal(size=11)
        ps.append(conformal.conformal_p(stat, 0))
    ps = np.array(ps)
    # mean p of an exchangeable candidate should be ~0.5
    assert 0.4 < ps.mean() < 0.6


def test_conformal_p_small_for_clear_winner():
    stat = np.array([5.0, 0.1, 0.0, -0.2, 0.05])
    assert conformal.conformal_p(stat, 0) <= conformal.conformal_p(stat, 1)
    assert conformal.conformal_p(stat, 0) == (1.0 + 0) / (5 + 1.0)


def test_merge_max_is_conservative():
    p_pop = np.array([0.1, 0.5, 0.9])
    p_sem = np.array([0.3, 0.2, 0.4])
    merged = conformal.merge_pvalues(p_pop, p_sem, mode="max")
    assert np.allclose(merged, np.array([0.3, 0.5, 0.9]))


def test_residualizer_removes_monotone_popularity():
    rng = np.random.default_rng(1)
    n = 60
    log_pop = np.linspace(0, 5, n)
    true_pref = rng.normal(size=n)
    # evidence = strong additive popularity shadow + preference
    evidence = 3.0 * log_pop + true_pref
    resid = residualizer.residualize_additive(
        evidence, {"log_pop": log_pop}, keys=("log_pop",)
    )
    # residual should correlate with preference, not popularity
    corr_pop = abs(np.corrcoef(resid, log_pop)[0, 1])
    corr_pref = abs(np.corrcoef(resid, true_pref)[0, 1])
    assert corr_pop < 0.3
    assert corr_pref > 0.6


def test_shrinkage_pulls_sparse_users_toward_target():
    stat = np.array([2.0, -2.0, 0.0, 1.0])
    unc = np.array([1.0, 1.0, 1.0, 1.0])
    facet = np.array([0, 0, 1, 1], dtype=float)
    cold = conformal.james_stein_shrink(stat, unc, facet, n_history=0, n_cap=20)
    rich = conformal.james_stein_shrink(stat, unc, facet, n_history=20, n_cap=20)
    # rich users: no shrink (identity); cold users: pulled toward bucket means
    assert np.allclose(rich, stat)
    assert np.abs(cold - stat).sum() > 0.0


def test_config_ablation_switches_independent():
    base = CCPaceConfig()
    text_only = CCPaceConfig(**{**base.to_dict(), "use_cf_tokens": False, "use_cf_nuisance": False})
    assert base.use_cf_tokens is True
    assert text_only.use_cf_tokens is False
    assert text_only.use_residualizer is True  # unaffected
