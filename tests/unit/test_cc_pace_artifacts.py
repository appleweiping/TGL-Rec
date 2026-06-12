"""Tests for CC-PACE data-prep seams: profiles, CF artifact loader, builders.

CPU-only except the SASRec builder end-to-end test, which skips without torch.
"""

from __future__ import annotations

import json

import pytest

from llm4rec.methods.cc_pace.cf_conditioning import (
    load_cf_artifacts,
    provider_from_artifacts,
)
from llm4rec.methods.cc_pace.text_facets import (
    coarse_category,
    extract_brand,
    profile_from_history,
)

BEAUTY_TITLES = [
    "OZNaturals Retinol Serum",
    "OZNaturals Vitamin C Facial Cleanser - Anti Aging Face Wash",
    "Pure Hyaluronic Acid Serum with Vitamin C for Face - Anti-Wrinkle Treatment",
    "L'ANGE HAIR Argan-Infused Round Brush | Tourmaline Ceramic Barrel",
]


def test_coarse_category_matches_beauty_lexicon():
    assert coarse_category("OZNaturals Retinol Serum") == "skincare"
    assert coarse_category("Volumizing Shampoo for Fine Hair") == "hair care"
    assert coarse_category("Matte Lipstick Set") == "makeup"
    assert coarse_category("Completely Unrelated Widget") == ""


def test_extract_brand_heuristics():
    assert extract_brand("OZNaturals Retinol Serum") == "OZNaturals"
    assert extract_brand("L'ANGE HAIR Argan-Infused Round Brush") == "L'ANGE HAIR"
    # ordinary leading words are not brands
    assert extract_brand("Pure Hyaluronic Acid Serum") == ""


def test_profile_from_history_slots():
    prof = profile_from_history(BEAUTY_TITLES, domain="beauty")
    assert any("skincare" in c for c in prof["top_categories"])
    assert any("OZNaturals" in b.lower() or "oznaturals" in b for b in prof["liked_brands"])
    assert any("anti-aging" in c for c in prof["concerns"])
    assert any("retinol" in i for i in prof["ingredient_prefs"])
    assert "price_band" not in prof  # no price data in task files


def test_profile_from_history_empty_when_no_signal():
    assert profile_from_history(["zzz qqq"], domain="beauty") == {}
    # unknown domain -> no lexicons -> at most brand slot
    prof = profile_from_history(BEAUTY_TITLES, domain="books")
    assert "top_categories" not in prof


def _tiny_artifact() -> dict:
    return {
        "domain": "beauty",
        "user_scores": {"u1": {"i1": 1.2, "i2": -0.3}},
        "item_neighbors": {"i1": ["t-i2", "t-i3"], "i2": ["t-i1"]},
        "item_clusters": {"i1": 0, "i2": 1},
        "item_popularity": {"i1": 5, "i2": 1},
        "item_category": {"i1": "skincare", "i2": ""},
    }


def test_artifact_loader_and_provider(tmp_path):
    path = tmp_path / "cf.json"
    path.write_text(json.dumps(_tiny_artifact()), encoding="utf-8")
    art = load_cf_artifacts(path)
    provider = provider_from_artifacts(art)

    ev = provider.evidence_tokens("u1", ["i1", "i2", "i9"])
    assert "also bought" in ev[0] and "affinity score=1.200" in ev[0]
    assert ev[2] == ""  # unknown item degrades gracefully

    nui = provider.nuisance("u1", ["i1", "i2", "i9"])
    assert nui["r_cf"].tolist() == [1.2, -0.3, 0.0]
    assert nui["cf_cluster"].tolist() == [0.0, 1.0, 0.0]

    # unknown user -> empty evidence, zero nuisance (never crashes)
    assert provider.evidence_tokens("u_unknown", ["i1"]) == [""]
    assert provider.nuisance("u_unknown", ["i1"])["r_cf"].tolist() == [0.0]


def test_artifact_loader_rejects_missing_keys(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"user_scores": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys"):
        load_cf_artifacts(path)


def test_provider_plugs_into_ranker(tmp_path):
    from llm4rec.methods.cc_pace.config import CCPaceConfig
    from llm4rec.rankers.base import RankingExample
    from llm4rec.rankers.cc_pace import CCPaceRanker

    art = {
        "domain": "beauty",
        "user_scores": {"u1": {f"i{j}": float(j) / 4 for j in range(8)}},
        "item_neighbors": {f"i{j}": [f"nbr of i{j}"] for j in range(8)},
        "item_clusters": {f"i{j}": j % 2 for j in range(8)},
    }
    path = tmp_path / "cf.json"
    path.write_text(json.dumps(art), encoding="utf-8")
    provider = provider_from_artifacts(load_cf_artifacts(path))

    items = [
        {"item_id": f"i{j}", "category": "skincare", "brand": f"b{j%3}",
         "keywords": f"serum {j}", "attrs": f"size {j}", "popularity": j}
        for j in range(8)
    ]
    ranker = CCPaceRanker(CCPaceConfig(n_label_randomizations=2), cf_provider=provider)
    ranker.fit([], items)
    ranker.set_profiles({"u1": {"concerns": "hydration", "top_categories": ["skincare"]}})
    res = ranker.rank(
        RankingExample(
            user_id="u1", history=["serum 3"], target_item="i3",
            candidate_items=[it["item_id"] for it in items], domain="beauty",
        )
    )
    assert len(res.items) == 8
    assert res.metadata["use_cf_tokens"] is True


def test_cf_builder_end_to_end(tmp_path):
    torch = pytest.importorskip("torch")
    del torch
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import build_cc_pace_cf_artifacts as builder
        import build_cc_pace_profiles as prof_builder
    finally:
        sys.path.remove(str(scripts_dir))

    # tiny synthetic domain: 6 users x 4-5 interactions over 10 items
    train_path = tmp_path / "train.jsonl"
    task_path = tmp_path / "task.jsonl"
    items = [f"B{j:03d}" for j in range(10)]
    rows = []
    for u in range(6):
        seq = [items[(u + k) % 10] for k in range(4)]
        for k, iid in enumerate(seq):
            rows.append({"user_id": f"u{u}", "item_id": iid, "timestamp": 1000.0 + k})
    train_path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    task_rows = []
    for u in range(6):
        cands = items[:6]
        task_rows.append(
            {
                "user_id": f"u{u}",
                "history": [f"Retinol Serum {k}" for k in range(3)],
                "history_item_ids": [items[(u + k) % 10] for k in range(3)],
                "candidate_item_ids": cands,
                "candidate_titles": [f"OZNaturals Serum {c}" for c in cands],
                "candidate_texts": [f"Title: OZNaturals Serum {c}" for c in cands],
                "positive_item_index": 0,
                "positive_item_id": cands[0],
                "positive_item_title": f"OZNaturals Serum {cands[0]}",
            }
        )
    task_path.write_text(
        "\n".join(json.dumps(r) for r in task_rows) + "\n", encoding="utf-8"
    )

    out = tmp_path / "cf_artifacts.json"
    builder.main(
        [
            "--train-interactions", str(train_path), "--task", str(task_path),
            "--out", str(out), "--epochs", "2", "--hidden-dim", "8",
            "--num-layers", "1", "--num-heads", "1", "--clusters", "2",
            "--neighbors", "2", "--batch-size", "4",
        ]
    )
    art = load_cf_artifacts(out)
    # items[9] never appears in any user's sequence -> 9 distinct train items
    assert art["provenance"]["vocab_size"] == 9
    assert len(art["user_scores"]) == 6
    assert all(len(v) == 2 for v in art["item_neighbors"].values())
    assert set(art["item_clusters"].values()) <= {0, 1}
    assert art["item_popularity"][items[0]] >= 1

    prof_out = tmp_path / "profiles.json"
    prof_builder.main(["--task", str(task_path), "--out", str(prof_out)])
    payload = json.loads(prof_out.read_text(encoding="utf-8"))
    assert payload["provenance"]["n_users_with_profile"] == 6
    assert "ingredient_prefs" in payload["profiles"]["u0"]

    # the produced artifacts drive the ranker end-to-end (mock judge)
    from llm4rec.methods.cc_pace.config import CCPaceConfig
    from llm4rec.rankers.base import RankingExample
    from llm4rec.rankers.cc_pace import CCPaceRanker

    provider = provider_from_artifacts(art)
    meta = [
        {"item_id": c, "keywords": f"OZNaturals Serum {c}", "attrs": "",
         "popularity": art["item_popularity"].get(c, 0)}
        for c in items[:6]
    ]
    ranker = CCPaceRanker(CCPaceConfig(n_label_randomizations=2), cf_provider=provider)
    ranker.fit([], meta)
    ranker.set_profiles(payload["profiles"])
    res = ranker.rank(
        RankingExample(
            user_id="u0", history=["Retinol Serum 0"], target_item=items[0],
            candidate_items=items[:6], domain="beauty",
        )
    )
    assert len(res.items) == 6
