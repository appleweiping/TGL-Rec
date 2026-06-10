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


def evaluate(ranker: CCPaceRanker, rows: list[dict]) -> dict:
    # index item metadata so the ranker can render schema fields
    item_records = []
    seen = set()
    for r in rows:
        for it in r["items"]:
            if it["item_id"] not in seen:
                item_records.append(it)
                seen.add(it["item_id"])
    ranker.fit([], item_records)

    n = len(rows)
    agg = {f"NDCG@{k}": 0.0 for k in (5, 10, 20)}
    agg.update({f"HR@{k}": 0.0 for k in (5, 10, 20)})
    agg["MRR"] = 0.0
    for r in rows:
        res = ranker.rank(r["example"])
        tgt = r["example"].target_item
        for k in (5, 10, 20):
            agg[f"NDCG@{k}"] += ndcg_at_k(res.items, tgt, k)
            agg[f"HR@{k}"] += hit_rate_at_k(res.items, tgt, k)
        agg["MRR"] += mrr_at_k(res.items, tgt, len(res.items))
    return {k: v / n for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="beauty ranking_test.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mock", action="store_true", help="CPU mock judge (plumbing/CI only)")
    ap.add_argument("--adapter", default="", help="path to trained judge LoRA (optional)")
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
        from llm4rec.methods.cc_pace.hf_judge import HFForcedChoiceModel

        model = HFForcedChoiceModel(args.adapter or cfg.backbone_model)

    rows = load_examples(args.task, args.limit or None)
    print(f"loaded {len(rows)} beauty examples; variant={args.variant} mock={args.mock}", flush=True)
    ranker = CCPaceRanker(cfg, model=model)
    metrics = evaluate(ranker, rows)

    result = {
        "variant": args.variant,
        "n_examples": len(rows),
        "metrics": metrics,
        "sota_bar": SOTA_BAR,
        "beats_sota_ndcg10": metrics["NDCG@10"] >= SOTA_BAR["NDCG@10"],
        "mock": args.mock,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result["metrics"], indent=2))
    print("SOTA bar NDCG@10 =", SOTA_BAR["NDCG@10"], "| ours =", round(metrics["NDCG@10"], 4))


if __name__ == "__main__":
    main()
