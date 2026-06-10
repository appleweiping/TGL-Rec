"""Tests for CC-PACE schema/ranker/trainer-loss (CPU, mock judge)."""

from __future__ import annotations

import random

import numpy as np

from llm4rec.methods.cc_pace import schema as schema_mod
from llm4rec.methods.cc_pace.config import CCPaceConfig


def _items(n):
    return [
        {"item_id": f"i{j}", "category": "Beauty > Skincare", "brand": f"b{j%3}",
         "keywords": f"serum {j}", "attrs": f"size {j}", "popularity": j}
        for j in range(n)
    ]


def test_render_panel_label_randomization_is_a_permutation():
    cfg = CCPaceConfig()
    items = _items(10)
    rng = random.Random(7)
    panel = schema_mod.render_panel(
        candidates=items, history_titles=["x", "y"], profile=None,
        cf_evidence=None, cfg=cfg, rng=rng,
    )
    assert sorted(panel.presentation_to_original) == list(range(10))
    assert len(panel.candidate_blocks) == 10
    # every original index maps to exactly one label
    assert len(set(panel.original_to_label.values())) == 10


def test_profile_slots_rendered_when_enabled():
    cfg = CCPaceConfig()
    block = schema_mod.render_user_block(
        ["a", "b"], {"concerns": "dryness", "liked_brands": ["x", "y"]}, cfg
    )
    assert "User profile:" in block
    assert "concerns: dryness" in block


def test_ranker_end_to_end_with_mock_model():
    from llm4rec.rankers.cc_pace import CCPaceRanker
    from llm4rec.rankers.base import RankingExample

    cfg = CCPaceConfig(n_label_randomizations=2)
    ranker = CCPaceRanker(cfg)
    items = _items(8)
    ranker.fit([], items)
    ex = RankingExample(
        user_id="u1",
        history=["serum 3", "size 3"],
        target_item="i3",
        candidate_items=[it["item_id"] for it in items],
        domain="beauty",
    )
    res = ranker.rank(ex)
    assert len(res.items) == 8
    assert set(res.items) == {it["item_id"] for it in items}
    assert len(res.scores) == 8
    assert "p_pop" in res.metadata


def test_plackett_luce_loss_and_grad():
    from llm4rec.trainers.cc_pace_trainer import (
        plackett_luce_top1_loss,
        plackett_luce_grad,
    )

    stat = np.array([2.0, 0.0, -1.0, 0.5])
    loss_good = plackett_luce_top1_loss(stat, positive_idx=0)
    loss_bad = plackett_luce_top1_loss(stat, positive_idx=2)
    assert loss_good < loss_bad  # positive already on top -> lower loss
    g = plackett_luce_grad(stat, positive_idx=0)
    assert g.shape == stat.shape
    assert g[0] < 0  # push positive up
