#!/usr/bin/env python3
"""CC-PACE beauty experiment driver (zero-shot probe + ablations).

Runs the CC-PACE ranker over the beauty same-candidate panel and reports
HR/NDCG/MRR vs the SOTA bar (promax NDCG@10=0.1506). Designed so ANY agent can:
  - run the zero-shot kill test (frozen Qwen3-8B, no training),
  - run the by-design ablations (text-only / no-residualizer / no-shrinkage),
  - and (once a LoRA adapter exists) point --adapter at it to evaluate the trained
    judge under the identical pipeline.

This driver is backbone-agnostic: with --mock it runs on CPU using the ranker's
mock judge (for plumbing/CI); on the server omit --mock to load Qwen3-8B.

Go/kill thresholds: see docs/method_v2_decision_CC-PACE.md.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys

# ensure src on path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm4rec.metrics.ranking import ndcg_at_k, hit_rate_at_k, mrr_at_k  # noqa: E402
from llm4rec.methods.cc_pace.config import CCPaceConfig  # noqa: E402
from llm4rec.rankers.base import RankingExample  # noqa: E402
from llm4rec.rankers.cc_pace import CCPaceRanker  # noqa: E402

SOTA_BAR = {"baseline": "promax", "NDCG@10": 0.1506, "NDCG@5": 0.1226, "MRR": 0.1429}


def parse_listish(x):
    if isinstance(x, list):
        return x
    if not x:
        return []
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []


def load_examples(task_path: str, limit: int | None):
    rows = []
    with open(task_path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                break
            d = json.loads(line)
            cand_ids = parse_listish(d.get("candidate_item_ids"))
            titles = parse_listish(d.get("candidate_titles"))
            texts = parse_listish(d.get("candidate_texts")) or titles
            pos_idx = int(d.get("positive_item_index", -1))
            if not cand_ids or not (0 <= pos_idx < len(cand_ids)):
                continue
            item_meta = []
            for j, cid in enumerate(cand_ids):
                item_meta.append(
                    {
                        "item_id": str(cid),
                        "keywords": titles[j] if j < len(titles) else "",
                        "attrs": texts[j] if j < len(texts) else "",
                    }
                )
            rows.append(
                {
                    "example": RankingExample(
                        user_id=str(d.get("user_id", i)),
                        history=[str(h) for h in parse_listish(d.get("history"))],
                        target_item=str(cand_ids[pos_idx]),
                        candidate_items=[str(c) for c in cand_ids],
                        domain="beauty",
                    ),
                    "items": item_meta,
                }
            )
    return rows


def evaluate(ranker: CCPaceRanker, rows: list[dict], *, partial_path: str = "") -> dict:
    """Evaluate with per-user checkpointing.

    Per-user metric rows stream to ``partial_path`` (.jsonl) as they complete;
    on restart, already-scored users are skipped and their rows reused. The
    per-user rows are also the input for paired-bootstrap significance tests.
    """
    # index item metadata so the ranker can render schema fields
    item_records = []
    seen = set()
    for r in rows:
        for it in r["items"]:
            if it["item_id"] not in seen:
                item_records.append(it)
                seen.add(it["item_id"])
    ranker.fit([], item_records)

    done: dict[str, dict] = {}
    if partial_path and os.path.exists(partial_path):
        with open(partial_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    done[d["user_id"]] = d
        print(f"resuming: {len(done)} users already scored in {partial_path}", flush=True)

    partial_fh = open(partial_path, "a", encoding="utf-8") if partial_path else None
    per_user: list[dict] = []
    try:
        for i, r in enumerate(rows):
            uid = r["example"].user_id
            if uid in done:
                per_user.append(done[uid])
                continue
            res = ranker.rank(r["example"])
            tgt = r["example"].target_item
            row = {"user_id": uid}
            for k in (5, 10, 20):
                row[f"NDCG@{k}"] = ndcg_at_k(res.items, tgt, k)
                row[f"HR@{k}"] = hit_rate_at_k(res.items, tgt, k)
            row["MRR"] = mrr_at_k(res.items, tgt, len(res.items))
            per_user.append(row)
            if partial_fh:
                partial_fh.write(json.dumps(row) + "\n")
                partial_fh.flush()
            if (i + 1) % 25 == 0:
                so_far = sum(x["NDCG@10"] for x in per_user) / len(per_user)
                print(f"[{i + 1}/{len(rows)}] running NDCG@10={so_far:.4f}", flush=True)
    finally:
        if partial_fh:
            partial_fh.close()

    n = max(1, len(per_user))
    keys = [f"{m}@{k}" for m in ("NDCG", "HR") for k in (5, 10, 20)] + ["MRR"]
    return {key: sum(x[key] for x in per_user) / n for key in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="beauty ranking_test.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mock", action="store_true", help="CPU mock judge (plumbing/CI only)")
    ap.add_argument(
        "--judge",
        choices=["hf", "vllm"],
        default="hf",
        help="forced-choice judge backend for non-mock runs",
    )
    ap.add_argument("--adapter", default="", help="path to trained judge LoRA (optional)")
    ap.add_argument(
        "--cf-artifacts",
        default="",
        help="CF artifact JSON from build_cc_pace_cf_artifacts.py (frozen SASRec signal)",
    )
    ap.add_argument(
        "--profiles",
        default="",
        help="profiles JSON from build_cc_pace_profiles.py (train-history profile slots)",
    )
    ap.add_argument(
        "--variant",
        default="full",
        choices=["full", "text_only", "no_residualizer", "no_shrinkage", "rich_residualizer"],
    )
    args = ap.parse_args()

    cfg_kwargs = {}
    if args.variant == "text_only":
        cfg_kwargs.update(use_cf_tokens=False, use_cf_nuisance=False)
    elif args.variant == "no_residualizer":
        cfg_kwargs.update(use_residualizer=False)
    elif args.variant == "no_shrinkage":
        cfg_kwargs.update(use_shrinkage=False)
    elif args.variant == "rich_residualizer":
        cfg_kwargs.update(residualizer_rich=True)
    cfg = CCPaceConfig(**{**CCPaceConfig().to_dict(), **cfg_kwargs})

    model = None
    if not args.mock:
        if args.judge == "hf":
            from llm4rec.methods.cc_pace.hf_judge import HFForcedChoiceModel

            model = HFForcedChoiceModel(args.adapter or cfg.backbone_model)
        else:
            from llm4rec.methods.cc_pace.vllm_judge import VLLMForcedChoiceModel

            model = VLLMForcedChoiceModel(args.adapter or cfg.backbone_model)

    cf_provider = None
    item_pop: dict = {}
    item_cat: dict = {}
    if args.cf_artifacts:
        from llm4rec.methods.cc_pace.cf_conditioning import (
            load_cf_artifacts,
            provider_from_artifacts,
        )

        art = load_cf_artifacts(args.cf_artifacts)
        cf_provider = provider_from_artifacts(art)
        item_pop = art.get("item_popularity", {})
        item_cat = art.get("item_category", {})
        print(f"CF artifacts: {len(art['user_scores'])} users scored", flush=True)

    rows = load_examples(args.task, args.limit or None)
    # enrich candidate meta so the residualizer's log_pop / facet_bucket are real
    if item_pop or item_cat:
        for r in rows:
            for it in r["items"]:
                iid = it["item_id"]
                if item_pop:
                    it["popularity"] = float(item_pop.get(iid, 0.0))
                if item_cat:
                    it["category"] = item_cat.get(iid, "")
    print(
        f"loaded {len(rows)} beauty examples; variant={args.variant} "
        f"mock={args.mock} judge={args.judge}",
        flush=True,
    )
    ranker = CCPaceRanker(cfg, model=model, cf_provider=cf_provider)
    if args.profiles:
        with open(args.profiles, encoding="utf-8") as fh:
            payload = json.load(fh)
        profiles = payload.get("profiles", payload)
        ranker.set_profiles(profiles)
        print(f"profiles: {len(profiles)} users", flush=True)
    partial = args.out + ".per_user.jsonl" if not args.mock else ""
    metrics = evaluate(ranker, rows, partial_path=partial)

    result = {
        "variant": args.variant,
        "n_examples": len(rows),
        "metrics": metrics,
        "sota_bar": SOTA_BAR,
        "beats_sota_ndcg10": metrics["NDCG@10"] >= SOTA_BAR["NDCG@10"],
        "mock": args.mock,
        "judge": args.judge if not args.mock else "mock",
        "cf_artifacts": bool(args.cf_artifacts),
        "profiles": bool(args.profiles),
        "adapter": args.adapter or None,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result["metrics"], indent=2))
    print("SOTA bar NDCG@10 =", SOTA_BAR["NDCG@10"], "| ours =", round(metrics["NDCG@10"], 4))


if __name__ == "__main__":
    main()
