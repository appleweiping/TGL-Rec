"""Unit tests for the PaRC ranker end-to-end on CPU (mock duel model)."""

from __future__ import annotations

import numpy as np

from llm4rec.methods.parc.config import PaRCConfig
from llm4rec.methods.parc.pairwise_prompt import PairwiseDuelModel
from llm4rec.methods.parc.ranker import PaRCRanker
from llm4rec.rankers.base import RankingExample


class _ScriptedDuelModel:
    """Returns s = pref[A] - pref[B] using item titles parsed from the prompt.

    pref maps title -> latent preference. The duel logit equals the preference
    difference (already antisymmetric), so symmetrization is a no-op here and the
    BT field recovers ``pref`` up to gauge.
    """

    def __init__(self, pref: dict[str, float]):
        self.pref = pref

    def duel_logit(self, prompt: str) -> float:
        body = prompt.split("Two candidate items:", 1)[-1]
        a = body.split("A:", 1)[-1].split("B:", 1)[0].strip().rstrip("…").strip()
        b = body.split("B:", 1)[-1].split("Given", 1)[0].strip().rstrip("…").strip()
        return float(self.pref.get(a, 0.0) - self.pref.get(b, 0.0))


def _make_example(n=12):
    cand = [f"item{i}" for i in range(n)]
    return RankingExample(
        user_id="u1",
        history=["hist_a", "hist_b"],
        target_item="item0",
        candidate_items=cand,
    )


def _item_records(n=12):
    return [{"item_id": f"item{i}", "title": f"item{i}"} for i in range(n)]


def test_ranker_runs_and_reports_diagnostics():
    n = 12
    ex = _make_example(n)
    pony = {ex.user_id: {f"item{i}": float(n - i) for i in range(n)}}  # item0 best
    ranker = PaRCRanker(PaRCConfig(), model=_ScriptedDuelModel({}), lam=0.5, seed=0)
    ranker.fit([], _item_records(n))
    ranker.set_pony_scores(pony)
    res = ranker.rank(ex)
    assert len(res.items) == n
    assert set(res.items) == {f"item{i}" for i in range(n)}
    # scores are descending (sorted by the ranker)
    assert all(res.scores[i] >= res.scores[i + 1] for i in range(n - 1))
    assert res.metadata["n_comparisons"] <= res.metadata["duel_budget"]
    assert "var_beta" in res.metadata


def test_ranker_lambda_zero_recovers_pony_order():
    n = 15
    ex = _make_example(n)
    # arbitrary (non-monotone) pony scores
    rng = np.random.default_rng(0)
    vals = rng.normal(size=n)
    pony = {ex.user_id: {f"item{i}": float(vals[i]) for i in range(n)}}
    # duel model that DISAGREES with pony, to prove lam=0 ignores beta
    ranker = PaRCRanker(PaRCConfig(), model=_ScriptedDuelModel({f"item{i}": float(-vals[i]) for i in range(n)}), lam=0.0)
    ranker.fit([], _item_records(n))
    ranker.set_pony_scores(pony)
    res = ranker.rank(ex)
    pony_order = sorted([f"item{i}" for i in range(n)], key=lambda c: (-pony[ex.user_id][c], c))
    assert res.items == pony_order, "lambda=0 must reproduce the pony order"


def test_ranker_lambda_positive_uses_duel_signal():
    """With lambda>0 and a flat pony anchor, the duel preference drives the order."""
    n = 10
    ex = _make_example(n)
    pony = {ex.user_id: {f"item{i}": 0.0 for i in range(n)}}  # flat anchor
    # duel model prefers higher-index items
    pref = {f"item{i}": float(i) for i in range(n)}
    ranker = PaRCRanker(PaRCConfig(standardize_beta=False), model=_ScriptedDuelModel(pref), lam=1.0)
    ranker.fit([], _item_records(n))
    ranker.set_pony_scores(pony)
    res = ranker.rank(ex)
    # with a flat anchor the order follows beta (the duel signal); the strongest
    # item (highest pref) surfaces at the top. The adaptive duel graph is sparse,
    # so we assert the top item rather than a full exact order.
    assert res.items[0] == "item9"
    # the duel-favoured items dominate the bottom-favoured ones at the top half
    top5 = set(res.items[:5])
    assert "item9" in top5 and "item8" in top5
    assert "item0" not in top5


def test_mock_model_satisfies_protocol():
    ranker = PaRCRanker()  # default _MockPairwiseDuelModel
    assert isinstance(ranker.model, PairwiseDuelModel)
