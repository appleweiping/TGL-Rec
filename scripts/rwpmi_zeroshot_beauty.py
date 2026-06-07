#!/usr/bin/env python3
"""RW-PMI zero-shot scorer for the beauty kill test (frozen Qwen3-8B).

Implements the tri-agent design decision in docs/redesign_decision_RW-PMI.md.
For each user event (history + 101 candidates) we compute four rankings:

  popularity     : rank by candidate_popularity_group (head>mid>tail), tie -> given order
  conditional    : rank by length-normalized  log p(cand_text | history)
  set_pmi        : conditional  -  log p(cand_text | null-user prompt)      [per-candidate]
  rw_pmi         : set_pmi + alpha * residual_witness_gain  - delta*log(1+toklen)

The witness gain uses ONLY history-derived probes (no future window -> leakage-free),
and is residualized against set_pmi across the whole eval set before being added.

This is the FROZEN-MODEL kill test: no training. Run, then read metrics vs the
beauty SOTA bar (proex NDCG@10=0.1506). Go/kill thresholds in the decision doc.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

POP_RANK = {"head": 2, "mid": 1, "tail": 0}


def parse_listish(x):
    """Fields in ranking_test.jsonl are python-repr strings of lists."""
    if isinstance(x, list):
        return x
    if x is None or x == "":
        return []
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []


@dataclass
class Event:
    user_id: str
    history_titles: list
    cand_texts: list
    cand_titles: list
    pop_groups: list
    positive_index: int
    n: int = field(init=False)

    def __post_init__(self):
        self.n = len(self.cand_texts)


def load_events(path: str, limit: int | None = None):
    events = []
    n_dropped = 0
    checked_schema = False
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                break
            d = json.loads(line)
            if not checked_schema:
                required = ["history", "candidate_texts", "candidate_titles",
                            "candidate_popularity_groups", "positive_item_index"]
                missing = [k for k in required if k not in d]
                if missing:
                    raise KeyError(
                        f"task file {path} missing expected keys {missing}; "
                        f"present keys = {sorted(d.keys())}"
                    )
                print(f"schema OK; keys = {sorted(d.keys())}", flush=True)
                checked_schema = True
            ev = Event(
                user_id=d.get("user_id", str(i)),
                history_titles=parse_listish(d.get("history")),
                cand_texts=parse_listish(d.get("candidate_texts")) or parse_listish(d.get("candidate_titles")),
                cand_titles=parse_listish(d.get("candidate_titles")),
                pop_groups=parse_listish(d.get("candidate_popularity_groups")),
                positive_index=int(d.get("positive_item_index", -1)),
            )
            ok = (ev.cand_texts and 0 <= ev.positive_index < len(ev.cand_texts)
                  and len(ev.pop_groups) == ev.n)
            if ok:
                events.append(ev)
            else:
                n_dropped += 1
    print(f"loaded {len(events)} events, dropped {n_dropped}", flush=True)
    if not events:
        raise RuntimeError("no usable events loaded — check task schema/paths")
    return events



# ----------------------------- scoring core ------------------------------- #

HISTORY_PROMPT = (
    "A user has purchased the following beauty products in order:\n{hist}\n"
    "Predict the next product the user will buy.\nNext product: "
)
NULL_PROMPT = (
    "Here is a beauty product available in the store.\nProduct: "
)
# leakage-free witness probe: derived from the user's own history, asked of the candidate
WITNESS_PROMPT = (
    "A user has purchased: {hist}\n"
    "This suggests the user's needs are: {witness}\n"
    "Considering that, the next product is: "
)


@torch.no_grad()
def seq_logprob(model, tok, prompt: str, target: str, device, max_ctx: int = 1024):
    """Length-normalized log p(target | prompt) under teacher forcing.

    Returns (sum_logprob, n_target_tokens). Caller decides normalization.
    """
    p_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=max_ctx).input_ids
    t_ids = tok(target, return_tensors="pt", truncation=True, max_length=256).input_ids
    if t_ids.shape[1] == 0:
        return -1e9, 1
    input_ids = torch.cat([p_ids, t_ids], dim=1).to(device)
    logits = model(input_ids).logits  # [1, L, V]
    # predict token at pos i from logits at i-1; target spans the last t_len positions
    t_len = t_ids.shape[1]
    logp = torch.log_softmax(logits[0, -t_len - 1 : -1, :].float(), dim=-1)
    tgt = input_ids[0, -t_len:]
    tok_lp = logp[torch.arange(t_len), tgt]
    return float(tok_lp.sum().item()), t_len


def make_witness(history_titles: list) -> str:
    """Leakage-free intent probe: a compact summary built ONLY from history.

    Deliberately simple/deterministic (no future, no learned extractor in the
    zero-shot test): the most recent few item titles as the user's 'needs'.
    The witness gain then measures whether a candidate is consistent with that.
    """
    recent = [t for t in history_titles[-4:] if t]
    if not recent:
        return "general beauty and personal care"
    return "; ".join(recent)


def score_event(model, tok, ev: Event, device, hist_max_items: int = 20):
    """Return dict of per-candidate score arrays for the four methods."""
    hist = "; ".join([t for t in ev.history_titles[-hist_max_items:] if t]) or "(no prior history)"
    hist_prompt = HISTORY_PROMPT.format(hist=hist)
    witness = make_witness(ev.history_titles)
    wit_prompt = WITNESS_PROMPT.format(hist=hist, witness=witness)

    cond = np.zeros(ev.n)        # log p(c | history), length-normalized
    marg = np.zeros(ev.n)        # log p(c | null user), length-normalized
    wit = np.zeros(ev.n)         # log p(c | history+witness), length-normalized
    toklen = np.zeros(ev.n)
    for j, ctext in enumerate(ev.cand_texts):
        ctext = (ctext or "").strip()[:512]
        s_c, n_c = seq_logprob(model, tok, hist_prompt, ctext, device)
        s_m, _ = seq_logprob(model, tok, NULL_PROMPT, ctext, device)
        s_w, _ = seq_logprob(model, tok, wit_prompt, ctext, device)
        cond[j] = s_c / max(n_c, 1)
        marg[j] = s_m / max(n_c, 1)
        wit[j] = s_w / max(n_c, 1)
        toklen[j] = n_c

    pop_buckets = np.array([POP_RANK.get(str(g).strip(), 0) for g in ev.pop_groups][: ev.n], dtype=float)
    # seeded jitter breaks intra-bucket ties uniformly at random (unbiased popularity anchor)
    rng = np.random.default_rng(abs(hash(ev.user_id)) % (2**32))
    pop = pop_buckets + rng.uniform(0, 0.5, size=ev.n)
    set_pmi = cond - marg
    raw_wit_gain = wit - cond  # does the witness probe lift this candidate beyond plain history?
    return {
        "popularity": pop,
        "conditional": cond,
        "set_pmi": set_pmi,
        "_raw_wit_gain": raw_wit_gain,
        "_toklen": toklen,
        "positive_index": ev.positive_index,
    }


# ----------------------------- metrics ------------------------------------ #

def rank_of_positive(scores: np.ndarray, pos_idx: int) -> int:
    """1-based rank of the positive under descending score (stable, ties -> worse rank)."""
    order = np.argsort(-scores, kind="stable")
    return int(np.where(order == pos_idx)[0][0]) + 1


def metrics_from_ranks(ranks: list) -> dict:
    ranks = np.array(ranks, dtype=float)
    out = {}
    for k in (5, 10, 20):
        hit = (ranks <= k).astype(float)
        out[f"HR@{k}"] = float(hit.mean())
        # NDCG with single relevant item: 1/log2(rank+1) if rank<=k else 0
        dcg = np.where(ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0)
        out[f"NDCG@{k}"] = float(dcg.mean())
    out["MRR"] = float((1.0 / ranks).mean())
    return out


def residualize(raw: np.ndarray, pmi: np.ndarray) -> np.ndarray:
    """Remove the linear component of raw explained by pmi (global least squares)."""
    if raw.std() == 0 or pmi.std() == 0:
        return raw - raw.mean()
    a = np.cov(raw, pmi, bias=True)[0, 1] / np.var(pmi)
    b = raw.mean() - a * pmi.mean()
    return raw - (a * pmi + b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="ranking_test.jsonl path")
    ap.add_argument("--model", default="/home/ajifang/models/Qwen/Qwen3-8B")
    ap.add_argument("--out", required=True, help="output json path")
    ap.add_argument("--limit", type=int, default=0, help="max events (0=all); use small for sanity")
    ap.add_argument("--alpha", type=float, default=1.0, help="witness-gain weight in rw_pmi")
    ap.add_argument("--delta", type=float, default=0.0, help="token-length penalty weight")
    args = ap.parse_args()

    events = load_events(args.task, args.limit or None)
    print(f"loaded {len(events)} events", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # left-truncate prompts so the trailing instruction cue is never cut for long histories
    tok.truncation_side = "left"
    if getattr(tok, "add_bos_token", False):
        # standalone target must not get a BOS (would pollute conditional baseline + length norm)
        tok.add_bos_token = False
        print("note: disabled tokenizer add_bos_token for clean target scoring", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    ).eval()

    per = []
    all_raw_wit, all_pmi = [], []
    for i, ev in enumerate(events):
        s = score_event(model, tok, ev, device)
        per.append(s)
        all_raw_wit.append(s["_raw_wit_gain"])
        all_pmi.append(s["set_pmi"])
        if (i + 1) % 50 == 0:
            print(f"  scored {i+1}/{len(events)}", flush=True)

    # global residualization of witness gain vs set_pmi (concat all candidates)
    flat_raw = np.concatenate(all_raw_wit)
    flat_pmi = np.concatenate(all_pmi)
    resid_flat = residualize(flat_raw, flat_pmi)
    corr = float(np.corrcoef(flat_raw, flat_pmi)[0, 1]) if flat_raw.std() and flat_pmi.std() else 0.0

    # rebuild per-event residual witness gain and assemble rankings
    methods = ["popularity", "conditional", "set_pmi", "rw_pmi"]
    ranks = {m: [] for m in methods}
    cur = 0
    for s in per:
        n = len(s["set_pmi"])
        rw_gain = resid_flat[cur : cur + n]
        cur += n
        rw = s["set_pmi"] + args.alpha * rw_gain - args.delta * np.log1p(s["_toklen"])
        score_map = {
            "popularity": s["popularity"],
            "conditional": s["conditional"],
            "set_pmi": s["set_pmi"],
            "rw_pmi": rw,
        }
        for m in methods:
            ranks[m].append(rank_of_positive(score_map[m], s["positive_index"]))

    result = {
        "n_events": len(per),
        "witness_pmi_corr_preresid": corr,  # gate: keep witness only if this < 0.80
        "alpha": args.alpha,
        "delta": args.delta,
        "metrics": {m: metrics_from_ranks(ranks[m]) for m in methods},
        "sota_bar_beauty": {"baseline": "proex", "NDCG@10": 0.1506, "NDCG@5": 0.1226, "MRR": 0.1429},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result["metrics"], indent=2))
    print("witness~pmi corr (pre-resid):", round(corr, 3))


if __name__ == "__main__":
    main()



