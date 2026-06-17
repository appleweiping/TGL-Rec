"""CPU unit tests for the PaRC M0 toys driver (dry-run wiring + manifest).

Asserts:
  - the pilot-manifest builder is deterministic (same seed -> same 1k draw) and
    disjoint from the official TEST users;
  - the M0 dry-run produces a verdict dict with the pre-registered keys and a
    KILL/PROCEED decision;
  - lambda=0 floor recovers pony NDCG@10 (PaRC is never worse than the floor in
    selection) and the results JSON is written.
All synthetic / mock — no server data, no GPU, no vllm.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np

# scripts/ is not on pythonpath (pyproject sets pythonpath=["src"]); add it so the
# M0 driver module is importable for these CPU wiring tests.
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

m0 = importlib.import_module("run_parc_m0_toys")


# --------------------------------------------------------------------------- #
# (a) pilot manifest: deterministic + disjoint
# --------------------------------------------------------------------------- #
def test_pilot_manifest_deterministic():
    valid = [f"u{i}" for i in range(5000)]
    a = m0.build_pilot_manifest(valid, set(), n=1000, seed=m0.PILOT_SEED)
    b = m0.build_pilot_manifest(valid, set(), n=1000, seed=m0.PILOT_SEED)
    assert a == b
    assert len(a) == 1000
    assert len(set(a)) == 1000  # no dupes


def test_pilot_manifest_disjoint_from_test():
    valid = [f"u{i}" for i in range(3000)]
    test_users = {f"u{i}" for i in range(2500, 3000)}  # overlap range
    pilot = m0.build_pilot_manifest(valid, test_users, n=1000, seed=m0.PILOT_SEED)
    assert set(pilot).isdisjoint(test_users)
    # all selected users are genuine validation users
    assert set(pilot).issubset(set(valid))


def test_pilot_manifest_different_seed_differs():
    valid = [f"u{i}" for i in range(5000)]
    a = m0.build_pilot_manifest(valid, set(), n=1000, seed=m0.PILOT_SEED)
    c = m0.build_pilot_manifest(valid, set(), n=1000, seed=m0.PILOT_SEED + 1)
    assert a != c


def test_pilot_manifest_caps_at_available():
    valid = [f"u{i}" for i in range(300)]
    pilot = m0.build_pilot_manifest(valid, set(), n=1000, seed=m0.PILOT_SEED)
    assert len(pilot) == 300


def test_write_pilot_manifest_schema(tmp_path):
    p = tmp_path / "pilot.csv"
    m0.write_pilot_manifest(p, ["u1", "u2"], seed=m0.PILOT_SEED, source="toys_valid")
    text = p.read_text(encoding="utf-8").splitlines()
    assert text[0] == "user_id,split,seed,source_panel"
    assert text[1].startswith("u1,pilot_heldout,")


# --------------------------------------------------------------------------- #
# (d) M0 dry-run verdict
# --------------------------------------------------------------------------- #
def test_m0_dry_run_produces_verdict(tmp_path):
    from llm4rec.methods.parc.config import PaRCConfig

    out = tmp_path / "m0.json"
    pilot = tmp_path / "pilot.csv"
    result = m0.run_m0(
        valid_task=tmp_path / "missing_valid.jsonl",   # absent -> synthetic panels
        pony_scores=tmp_path / "missing_pony.csv",      # absent -> synthetic pony
        test_task=None,
        pilot_csv=pilot,
        out_json=out,
        cfg=PaRCConfig(),
        run_gpu=False,
        limit=40,
        seed=0,
    )
    # pre-registered verdict keys
    for key in (
        "verdict", "paper_path", "reason", "lambda_selected", "lambda_collapsed",
        "ndcg10_pony", "ndcg10_parc", "validation_lift", "significant",
        "noninferiority_eps", "var_beta_mean", "cyclic_triple_rate_mean",
    ):
        assert key in result, f"missing verdict key {key}"
    assert result["verdict"] in ("KILL", "PROCEED")
    assert result["mode"] == "cpu_dry_run"
    assert result["duel_extraction"] == "cpu_mock_dry_run"
    assert result["noninferiority_eps"] == 0.002
    # results JSON written + reloadable
    assert out.exists()
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded["verdict"] == result["verdict"]
    # pilot manifest written
    assert pilot.exists()


def test_m0_dry_run_pilot_disjoint_flag(tmp_path):
    from llm4rec.methods.parc.config import PaRCConfig

    result = m0.run_m0(
        valid_task=tmp_path / "missing_valid.jsonl",
        pony_scores=tmp_path / "missing_pony.csv",
        test_task=None,
        pilot_csv=tmp_path / "pilot.csv",
        out_json=tmp_path / "m0.json",
        cfg=PaRCConfig(),
        run_gpu=False,
        limit=24,
        seed=1,
    )
    assert result["pilot_disjoint_from_test"] is True
    assert result["pilot_n"] >= 1


def test_m0_verdict_kill_on_zero_lift():
    """A duel model with no signal (flat beta) must collapse lambda -> KILL."""
    from llm4rec.methods.parc.config import PaRCConfig

    cfg = PaRCConfig()
    rng = np.random.default_rng(0)
    # synth panel results where beta is ~0 -> mixture never beats pony.
    panel_results = []
    for u in range(30):
        k = 12
        pony = rng.normal(size=k)
        beta = np.zeros(k)
        panel_results.append(
            {
                "user_id": f"u{u}",
                "pony": pony,
                "beta": beta,
                "pos_index": int(rng.integers(0, k)),
                "diag": {"order_var_fraction_beta": 0.0, "pairs": []},
            }
        )
    verdict = m0.m0_verdict(panel_results, cfg)
    assert verdict["verdict"] == "KILL"
    assert verdict["paper_path"] == "diagnostic_negative"
    # lambda=0 floor: PaRC NDCG@10 equals pony NDCG@10 when beta is inert.
    assert abs(verdict["ndcg10_parc"] - verdict["ndcg10_pony"]) < 1e-9


def test_m0_injected_batched_model_path(tmp_path):
    """Inject the offline VLLMDuelModel (batched) -> driver uses the batch path."""
    from llm4rec.methods.parc.config import PaRCConfig
    from llm4rec.methods.parc.vllm_duel_model import VLLMDuelModel

    model = VLLMDuelModel(offline=True)
    result = m0.run_m0(
        valid_task=tmp_path / "missing_valid.jsonl",
        pony_scores=tmp_path / "missing_pony.csv",
        test_task=None,
        pilot_csv=tmp_path / "pilot.csv",
        out_json=tmp_path / "m0.json",
        cfg=PaRCConfig(),
        run_gpu=False,
        limit=20,
        seed=0,
        model=model,
    )
    assert result["verdict"] in ("KILL", "PROCEED")
    assert "offline" in result["duel_extraction"].lower()


def test_load_panels_and_pony_roundtrip(tmp_path):
    """load_panels + load_pony_scores parse the frozen jsonl / scores.csv schemas."""
    task = tmp_path / "ranking_test.jsonl"
    rows = [
        {
            "user_id": "uA",
            "history": ["h1", "h2"],
            "candidate_titles": ["t0", "t1", "t2"],
            "candidate_item_ids": ["i0", "i1", "i2"],
            "positive_item_index": 1,
            "source_event_id": "e_uA",
        }
    ]
    task.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    panels = m0.load_panels(task)
    assert len(panels) == 1
    assert panels[0]["candidate_ids"] == ["i0", "i1", "i2"]
    assert panels[0]["pos_index"] == 1

    scores = tmp_path / "scores.csv"
    scores.write_text(
        "source_event_id,user_id,item_id,score\n"
        "e_uA,uA,i0,0.1\ne_uA,uA,i1,0.9\ne_uA,uA,i2,0.3\n",
        encoding="utf-8",
    )
    pony = m0.load_pony_scores(scores)
    assert pony["uA"]["i1"] == 0.9
    assert set(pony["uA"]) == {"i0", "i1", "i2"}
