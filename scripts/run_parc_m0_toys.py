#!/usr/bin/env python3
"""PaRC M0 driver (pilot = toys) — the Block-0 kill-gate runner.

Implements the M0 block of ``refine-logs/EXPERIMENT_PLAN_PaRC.md`` with the
STRICT no-test-leakage protocol: ALL proceed/kill + lambda/alpha selection happen
on the toys VALIDATION panel + a fixed 1,000-user held-out pilot subset
(``refine-logs/parc_m0_pilot_users.csv``, seed 20260506, disjoint from the 10k
official TEST users). The official TEST panel is NOT touched here.

Stages
------
(a) ``build_pilot_manifest`` — deterministic 1,000-user subset of the toys
    validation users, drawn seed=20260506, verified disjoint from the official
    TEST user ids. Written to ``refine-logs/parc_m0_pilot_users.csv``.
(b) ``load_pony_scores`` — pony's toys Qwen pointwise posteriors
    (``outputs/toys_large10000_100neg_ccrp_v3/scores.csv``) as the alpha-anchor
    (``--pony-scores`` overridable).
(c) PaRCRanker over the validation panels with a duel model:
      - DEFAULT dry-run: CPU, mock duel model, validates the FULL wiring (manifest
        -> pony anchor -> duels -> BT field -> lambda-mixture -> verdict) with NO
        GPU/vLLM. The actual vLLM run is GPU and launched later with --run.
      - --run: load ``VLLMDuelModel`` (Qwen3-8B) and route the whole duel schedule
        through ONE batched ``duel_logits_batch`` per user (PaRC's throughput
        advantage). Imports vllm only under --run.
(d) On validation: select lambda; report PaRC-vs-pony NDCG@10 with paired
    significance + non-inferiority (eps=0.002), Var(beta), intransitivity; emit
    the pre-registered KILL/PROCEED verdict (lambda->0 OR no significant lift =>
    KILL). Results JSON under ``refine-logs/``.

Run examples
------------
    # CPU dry-run (default): validates wiring, mock duels, no GPU.
    python scripts/run_parc_m0_toys.py --limit 50

    # GPU run (server): real Qwen3-8B duels via vLLM.
    python scripts/run_parc_m0_toys.py --run \
        --valid-task .../toys_..._valid_same_candidate/ranking_test.jsonl \
        --pony-scores .../outputs/toys_large10000_100neg_ccrp_v3/scores.csv \
        --test-users .../toys_..._test_same_candidate/ranking_test.jsonl
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ensure src on path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm4rec.evaluation.significance import paired_randomization_test  # noqa: E402
from llm4rec.methods.parc import bt_field as bt_mod  # noqa: E402
from llm4rec.methods.parc import calibrate as calib_mod  # noqa: E402
from llm4rec.methods.parc import duels as duels_mod  # noqa: E402
from llm4rec.methods.parc import pairwise_prompt as pp_mod  # noqa: E402
from llm4rec.methods.parc.config import PaRCConfig  # noqa: E402

PILOT_SEED = 20260506
PILOT_N = 1000
NONINF_EPS = 0.002          # absolute NDCG@10 non-inferiority margin (pre-set)
PROCEED_LIFT = 0.005        # validation lift threshold for the method-success path
COLLAPSE_TOL = 1e-9         # lambda <= tol counts as collapsed-to-zero

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALID_TASK = (
    REPO_ROOT
    / "outputs/baselines/external_tasks/toys_large10000_100neg_valid_same_candidate/ranking_test.jsonl"
)
DEFAULT_PONY_SCORES = REPO_ROOT / "outputs/toys_large10000_100neg_ccrp_v3/scores.csv"
DEFAULT_PILOT_CSV = REPO_ROOT / "refine-logs/parc_m0_pilot_users.csv"


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def _parse_listish(x: Any) -> list:
    if isinstance(x, list):
        return x
    if not x:
        return []
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []


def load_panels(task_path: str | Path, limit: int | None = None) -> list[dict]:
    """Load same-candidate ranking panels from a ``ranking_test.jsonl`` file.

    Each returned panel dict:
      ``{"user_id", "history": [str], "candidate_ids": [str],
         "titles": [str], "pos_index": int}``.
    """
    rows: list[dict] = []
    with open(task_path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit and len(rows) >= limit:
                break
            d = json.loads(line)
            cand_ids = [str(c) for c in _parse_listish(d.get("candidate_item_ids"))]
            titles = [str(t) for t in _parse_listish(d.get("candidate_titles"))]
            pos_idx = int(d.get("positive_item_index", -1))
            if not cand_ids or not (0 <= pos_idx < len(cand_ids)):
                continue
            if len(titles) < len(cand_ids):
                titles = titles + cand_ids[len(titles):]
            rows.append(
                {
                    "user_id": str(d.get("user_id", i)),
                    "history": [str(h) for h in _parse_listish(d.get("history"))],
                    "candidate_ids": cand_ids,
                    "titles": titles,
                    "pos_index": pos_idx,
                }
            )
    return rows


def load_pony_scores(scores_csv: str | Path) -> dict[str, dict[str, float]]:
    """Load pony posteriors as ``{user_id -> {item_id -> score}}``.

    Schema: ``source_event_id, user_id, item_id, score`` (frozen pony scores.csv).
    """
    pony: dict[str, dict[str, float]] = {}
    with open(scores_csv, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            uid = str(r["user_id"])
            pony.setdefault(uid, {})[str(r["item_id"])] = float(r["score"])
    return pony


def load_test_user_ids(test_task: str | Path | None) -> set[str]:
    """Official TEST user ids (for the disjointness guarantee). Empty if unavailable."""
    if not test_task or not Path(test_task).exists():
        return set()
    ids: set[str] = set()
    with open(test_task, encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "user_id" in d:
                ids.add(str(d["user_id"]))
    return ids


# --------------------------------------------------------------------------- #
# (a) pilot-split manifest
# --------------------------------------------------------------------------- #
def build_pilot_manifest(
    valid_user_ids: list[str],
    test_user_ids: set[str],
    *,
    n: int = PILOT_N,
    seed: int = PILOT_SEED,
) -> list[str]:
    """Deterministic 1,000-user pilot subset drawn from validation users.

    Drawn with ``numpy.random.default_rng(seed)`` over the SORTED unique
    validation user ids (so the draw is reproducible regardless of file order),
    after removing any official TEST users (disjointness is enforced, not merely
    assumed). Returns the selected user ids in draw order.
    """
    eligible = sorted({str(u) for u in valid_user_ids} - {str(t) for t in test_user_ids})
    rng = np.random.default_rng(seed)
    take = min(n, len(eligible))
    idx = rng.choice(len(eligible), size=take, replace=False)
    return [eligible[int(i)] for i in idx]


def write_pilot_manifest(path: str | Path, user_ids: list[str], *, seed: int, source: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["user_id", "split", "seed", "source_panel"])
        for uid in user_ids:
            w.writerow([uid, "pilot_heldout", seed, source])


# --------------------------------------------------------------------------- #
# (c) PaRC over panels (CPU mock dry-run OR batched vLLM)
# --------------------------------------------------------------------------- #
def _schedule_pairs(pony_vec: np.ndarray, cfg: PaRCConfig, seed: int) -> list[tuple[int, int]]:
    """Replay the (data-independent) adaptive schedule to collect the (i,j) pairs.

    The schedule depends only on the pony order + RNG (NOT on duel outcomes), so we
    can enumerate the exact pairs once and then evaluate them in ONE batch.
    """
    recorded: list[tuple[int, int]] = []

    def record(i: int, j: int) -> float:
        recorded.append((i, j))
        return 0.0

    duels_mod.run_duels(pony_vec, record, cfg, seed=seed)
    return recorded


def _beta_for_panel(
    panel: dict,
    pony_vec: np.ndarray,
    cfg: PaRCConfig,
    *,
    model: Any,
    seed: int,
    batched: bool,
) -> tuple[np.ndarray, dict]:
    """Run duels for one panel and fit the BT field; return ``(beta, diagnostics)``.

    ``batched=True`` (vLLM path): collect every duel prompt for the panel and run
    ONE ``duel_logits_batch`` so vLLM batches the whole schedule. Symmetrization
    needs both A/B orders, so each scheduled pair contributes two prompts.
    ``batched=False``: per-pair ``symmetrized_duel`` (mock CPU path).
    """
    titles = panel["titles"]
    history = panel["history"]

    if batched and hasattr(model, "duel_logits_batch"):
        pairs = _schedule_pairs(pony_vec, cfg, seed)
        prompts: list[str] = []
        for (i, j) in pairs:
            prompts.append(pp_mod.build_pairwise_prompt(history, titles[i], titles[j], cfg))
            if cfg.symmetrize:
                prompts.append(pp_mod.build_pairwise_prompt(history, titles[j], titles[i], cfg))
        logits = model.duel_logits_batch(prompts)
        # stitch back into symmetrized s_ij keyed by pair index
        s_by_pair: dict[tuple[int, int], float] = {}
        c = 0
        for (i, j) in pairs:
            lij = float(logits[c]); c += 1
            if cfg.symmetrize:
                lji = float(logits[c]); c += 1
                s_by_pair[(i, j)] = 0.5 * (lij - lji)
            else:
                s_by_pair[(i, j)] = lij
        pair_list = [(i, j, s_by_pair[(i, j)]) for (i, j) in pairs]
        n_comp = len(pairs)
        budget = duels_mod.budget_for(len(titles), cfg)
    else:
        def eval_fn(i: int, j: int) -> float:
            return pp_mod.symmetrized_duel(history, titles[i], titles[j], model, cfg)["s_ij"]

        sched = duels_mod.run_duels(pony_vec, eval_fn, cfg, seed=seed)
        pair_list = sched.as_pair_list()
        n_comp = sched.n_comparisons
        budget = sched.budget

    bt = bt_mod.fit_bt_field(
        pair_list, pony_vec, n_items=len(titles), l2_beta=cfg.bt_l2_beta,
        fit_alpha=cfg.bt_fit_alpha, anchor=cfg.anchor_pony,
    )
    diag = {
        "n_comparisons": n_comp,
        "duel_budget": budget,
        "alpha": float(bt.alpha),
        "var_beta": float(bt.var_beta),
        "order_var_fraction_beta": float(bt.order_var_fraction_beta),
        "pairs": pair_list,
    }
    return bt.beta, diag


def run_parc_on_panels(
    panels: list[dict],
    pony: dict[str, dict[str, float]],
    cfg: PaRCConfig,
    *,
    model: Any,
    seed: int,
    batched: bool,
) -> list[dict]:
    """Compute per-panel ``{pony, beta, pos_index, diag}`` for lambda selection.

    Panels with no pony coverage are skipped (the harness guarantees coverage on
    the real run; the dry-run mock supplies a flat anchor so wiring still flows).
    """
    out: list[dict] = []
    for panel in panels:
        uid = panel["user_id"]
        cand = panel["candidate_ids"]
        pdict = pony.get(uid, {})
        pony_vec = np.array([float(pdict.get(c, 0.0)) for c in cand], dtype=float)
        beta, diag = _beta_for_panel(
            panel, pony_vec, cfg, model=model, seed=seed, batched=batched
        )
        out.append(
            {
                "user_id": uid,
                "pony": pony_vec,
                "beta": beta,
                "pos_index": panel["pos_index"],
                "diag": diag,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# (d) verdict
# --------------------------------------------------------------------------- #
def m0_verdict(
    panel_results: list[dict],
    cfg: PaRCConfig,
    *,
    sig_rounds: int = 1000,
    sig_seed: int = PILOT_SEED,
) -> dict:
    """Select lambda on validation and emit the pre-registered KILL/PROCEED verdict.

    Verdict rule (Block 0 method-success path):
      PROCEED iff lift >= PROCEED_LIFT AND lambda not collapsed AND paired
      significance p<0.05. Otherwise KILL (-> diagnostic/negative path).
    Also reports the TEST-claim non-inferiority margin (eps) for downstream use
    and Var(beta) + intransitivity aggregates.
    """
    sel = calib_mod.select_lambda(
        [{"pony": r["pony"], "beta": r["beta"], "pos_index": r["pos_index"]} for r in panel_results],
        lambda_grid=cfg.lambda_grid,
        k=10,
        standardize_beta=cfg.standardize_beta,
    )

    # per-user NDCG@10 at lambda=0 (pony) and at the selected lambda (PaRC).
    pony_per_user, parc_per_user = [], []
    for r in panel_results:
        pony_s = calib_mod.mixture_score(r["pony"], r["beta"], 0.0, standardize_beta=cfg.standardize_beta)
        parc_s = calib_mod.mixture_score(r["pony"], r["beta"], sel.lam, standardize_beta=cfg.standardize_beta)
        pony_per_user.append(calib_mod.ndcg_at_k(pony_s, r["pos_index"], 10))
        parc_per_user.append(calib_mod.ndcg_at_k(parc_s, r["pos_index"], 10))

    sig = paired_randomization_test(pony_per_user, parc_per_user, num_rounds=sig_rounds, seed=sig_seed)
    p_value = sig.get("p_value")
    significant = bool(p_value is not None and p_value < 0.05)

    # intransitivity aggregate over panels that have a usable duel graph (full or
    # dense enough). Cheap aggregate: mean cyclic-triple rate over panels.
    cyc_rates = []
    for r in panel_results:
        pairs = r["diag"].get("pairs", [])
        n_items = len(r["pony"])
        if len(pairs) >= 3 and n_items >= 3:
            cyc_rates.append(
                calib_mod.cyclic_triple_rate(pairs, n_items, sample=cfg.cyclic_triples_sample or 0)
            )
    var_betas = [float(np.var(r["beta"])) for r in panel_results]
    order_fracs = [float(r["diag"]["order_var_fraction_beta"]) for r in panel_results]

    lift = float(sel.lift)
    collapsed = bool(sel.lam <= COLLAPSE_TOL)
    proceed = bool((lift >= PROCEED_LIFT) and (not collapsed) and significant)

    if proceed:
        verdict = "PROCEED"
        path = "method_success"
        reason = (
            f"validation lift {lift:.4f} >= {PROCEED_LIFT}, lambda={sel.lam} (not collapsed), "
            f"paired p={p_value:.4g} < 0.05"
        )
    else:
        verdict = "KILL"
        path = "diagnostic_negative"
        bits = []
        if collapsed:
            bits.append("lambda collapsed to ~0")
        if lift < PROCEED_LIFT:
            bits.append(f"lift {lift:.4f} < {PROCEED_LIFT}")
        if not significant:
            bits.append(f"no significant validation lift (p={p_value})")
        reason = "; ".join(bits) or "kill criteria met"

    return {
        "verdict": verdict,
        "paper_path": path,
        "reason": reason,
        "lambda_selected": float(sel.lam),
        "lambda_collapsed": collapsed,
        "lambda_curve": [[float(l), float(m)] for l, m in sel.curve],
        "ndcg10_pony": float(np.mean(pony_per_user)) if pony_per_user else 0.0,
        "ndcg10_parc": float(np.mean(parc_per_user)) if parc_per_user else 0.0,
        "validation_lift": lift,
        "paired_test": sig,
        "significant": significant,
        "noninferiority_eps": NONINF_EPS,
        "noninferior_to_pony": bool(
            (np.mean(parc_per_user) if parc_per_user else 0.0)
            >= (np.mean(pony_per_user) if pony_per_user else 0.0) - NONINF_EPS
        ),
        "var_beta_mean": float(np.mean(var_betas)) if var_betas else 0.0,
        "order_var_fraction_beta_mean": float(np.mean(order_fracs)) if order_fracs else 0.0,
        "cyclic_triple_rate_mean": float(np.mean(cyc_rates)) if cyc_rates else 0.0,
        "n_panels": len(panel_results),
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_m0(
    *,
    valid_task: str | Path,
    pony_scores: str | Path,
    test_task: str | Path | None,
    pilot_csv: str | Path,
    out_json: str | Path,
    cfg: PaRCConfig,
    run_gpu: bool,
    limit: int | None,
    seed: int = 0,
    model: Any | None = None,
) -> dict:
    """Full M0 orchestration. Returns the results dict (also written to out_json).

    ``model``: inject a duel model for tests. Default = CPU mock (dry-run) or
    ``VLLMDuelModel`` when ``run_gpu`` (imported lazily under --run).
    """
    valid_exists = Path(valid_task).exists()
    pony_exists = Path(pony_scores).exists()

    # ---- panels + pony anchor ----
    if valid_exists:
        panels = load_panels(valid_task, limit=limit)
    else:
        # dry-run without server data: synth a tiny deterministic panel set so the
        # wiring (manifest -> duels -> BT -> verdict) is still exercised on CPU.
        panels = _synthetic_panels(limit or 24, seed=seed)
    pony = load_pony_scores(pony_scores) if pony_exists else _synthetic_pony(panels)

    # ---- (a) pilot manifest (disjoint from official TEST) ----
    test_users = load_test_user_ids(test_task)
    valid_users = [p["user_id"] for p in panels]
    pilot_users = build_pilot_manifest(valid_users, test_users, n=PILOT_N, seed=PILOT_SEED)
    write_pilot_manifest(
        pilot_csv, pilot_users, seed=PILOT_SEED, source=str(valid_task)
    )
    disjoint = set(pilot_users).isdisjoint(test_users)

    # ---- (c) duel model ----
    batched = False
    if model is None:
        if run_gpu:
            from llm4rec.methods.parc.vllm_duel_model import VLLMDuelModel  # noqa: PLC0415

            model = VLLMDuelModel(config=cfg, seed=seed)
            batched = True
            extraction = getattr(model, "extraction", "vllm")
        else:
            from llm4rec.methods.parc.ranker import _MockPairwiseDuelModel  # noqa: PLC0415

            model = _MockPairwiseDuelModel()
            extraction = "cpu_mock_dry_run"
    else:
        batched = hasattr(model, "duel_logits_batch")
        extraction = getattr(model, "extraction", "injected")

    panel_results = run_parc_on_panels(
        panels, pony, cfg, model=model, seed=seed, batched=batched
    )

    # ---- (d) verdict ----
    verdict = m0_verdict(panel_results, cfg)

    result = {
        "milestone": "M0",
        "pilot_domain": "toys",
        "mode": "gpu_run" if run_gpu else "cpu_dry_run",
        "duel_extraction": extraction,
        "valid_task": str(valid_task),
        "valid_task_present": valid_exists,
        "pony_scores": str(pony_scores),
        "pony_scores_present": pony_exists,
        "pilot_manifest": str(pilot_csv),
        "pilot_n": len(pilot_users),
        "pilot_seed": PILOT_SEED,
        "pilot_disjoint_from_test": bool(disjoint),
        "n_test_users_checked": len(test_users),
        "config": cfg.to_dict(),
        **verdict,
    }
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=_json_default)
    return result


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# --------------------------------------------------------------------------- #
# synthetic CPU fixtures (dry-run only when server data absent)
# --------------------------------------------------------------------------- #
def _synthetic_panels(n_users: int, *, seed: int = 0, n_cand: int = 12) -> list[dict]:
    rng = np.random.default_rng(seed)
    panels = []
    for u in range(n_users):
        cand = [f"item{u}_{c}" for c in range(n_cand)]
        titles = [f"title token{c % 5} token{(c + u) % 7}" for c in range(n_cand)]
        panels.append(
            {
                "user_id": f"valid_user_{u}",
                "history": [f"token{int(rng.integers(0, 5))}" for _ in range(3)],
                "candidate_ids": cand,
                "titles": titles,
                "pos_index": int(rng.integers(0, n_cand)),
            }
        )
    return panels


def _synthetic_pony(panels: list[dict]) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(0)
    pony: dict[str, dict[str, float]] = {}
    for p in panels:
        pony[p["user_id"]] = {c: float(rng.normal()) for c in p["candidate_ids"]}
    return pony


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true",
                    help="GPU: load Qwen3-8B via vLLM and run real batched duels. "
                         "DEFAULT off = CPU dry-run that validates wiring with the mock.")
    ap.add_argument("--valid-task", default=str(DEFAULT_VALID_TASK),
                    help="toys VALIDATION ranking_test.jsonl")
    ap.add_argument("--pony-scores", default=str(DEFAULT_PONY_SCORES),
                    help="pony toys Qwen posteriors scores.csv (alpha-anchor)")
    ap.add_argument("--test-users", default="",
                    help="official TEST ranking_test.jsonl (for disjointness check; not scored)")
    ap.add_argument("--pilot-csv", default=str(DEFAULT_PILOT_CSV))
    ap.add_argument("--out", default=str(REPO_ROOT / "refine-logs/parc_m0_toys_result.json"))
    ap.add_argument("--limit", type=int, default=0, help="cap panels (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main(argv: list[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    cfg = PaRCConfig()
    result = run_m0(
        valid_task=args.valid_task,
        pony_scores=args.pony_scores,
        test_task=(args.test_users or None),
        pilot_csv=args.pilot_csv,
        out_json=args.out,
        cfg=cfg,
        run_gpu=args.run,
        limit=(args.limit or None),
        seed=args.seed,
    )
    print(json.dumps(
        {k: result[k] for k in (
            "milestone", "mode", "duel_extraction", "verdict", "paper_path", "reason",
            "lambda_selected", "ndcg10_pony", "ndcg10_parc", "validation_lift",
            "significant", "pilot_n", "pilot_disjoint_from_test",
        )},
        indent=2,
    ))
    return result


if __name__ == "__main__":
    main()
