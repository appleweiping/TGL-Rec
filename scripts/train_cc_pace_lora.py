#!/usr/bin/env python3
"""CC-PACE judge LoRA training (server/GPU): listwise PL over train-built panels.

Runs ONLY after the zero-shot GO (docs/HOW_TO_RUN_CC_PACE.md): NDCG@10 >= 0.13
and full > text_only. Trains the judge LoRA with the Plackett-Luce top-1 loss
(+ differentiable dCor popularity penalty) from llm4rec.trainers.cc_pace_trainer.

Tractability: the PL loss needs all 101 candidate scores in ONE autograd graph.
Scoring every multi-token label with gradients is infeasible, so training uses
the single-token surrogate: one forward of (panel prompt + "[") and the 101
scores are the log-softmax row at each label's DIGIT token id ("[" prefix is
shared by all labels and "]" is near-deterministic given the digits, so the
digit token carries the discriminating mass). Inference keeps the spec'd
full-label length-normalized scoring (hf_judge.py, equivalence-tested).

Leakage discipline (docs/method_v2_decision_CC-PACE.md):
  - Train panels only: pseudo-held-out positive = the user's LAST train item;
    history/profile come from the PREFIX only (the positive never enters the
    user block).
  - CF evidence for train panels comes from a PREFIX-trained SASRec (each
    user's last train item excluded), so CF memorization of the pseudo-positive
    cannot leak into the judge's training signal. Test-time CF artifacts are
    unchanged.
  - Negatives: popularity-matched sampling from the train vocab.
  - SPLIT CONFORMAL: panels are split into fold A (gradient steps) and fold B
    (held for conformal calibration, never trained on); the assignment is saved.

Usage (server):
  python scripts/train_cc_pace_lora.py \
      --train-interactions data/domains/beauty/train_interactions.jsonl \
      --cf-artifacts outputs/cc_pace_beauty/cf_artifacts.json \
      --out outputs/cc_pace_beauty/lora \
      --model-path /home/ajifang/models/Qwen/Qwen3-8B
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from llm4rec.methods.cc_pace.config import CCPaceConfig, load_cc_pace_config  # noqa: E402
from llm4rec.methods.cc_pace import schema as schema_mod  # noqa: E402
from llm4rec.methods.cc_pace.cf_conditioning import (  # noqa: E402
    PrecomputedCFProvider,
    load_cf_artifacts,
)
from llm4rec.methods.cc_pace.text_facets import profile_from_history  # noqa: E402
from llm4rec.trainers.cc_pace_trainer import build_training_plan  # noqa: E402

from build_cc_pace_cf_artifacts import (  # noqa: E402  (sibling script import)
    collect_titles,
    read_jsonl,
    train_sasrec,
)


def build_user_prefix_sequences(train_rows: list[dict]) -> dict[str, list[str]]:
    by_user: dict[str, list[tuple[float, str]]] = {}
    for r in train_rows:
        by_user.setdefault(str(r["user_id"]), []).append(
            (float(r.get("timestamp") or -1), str(r["item_id"]))
        )
    return {u: [i for _, i in sorted(v)] for u, v in by_user.items()}


def torch_dcor(a, b):
    """Differentiable distance correlation (mirrors cc_pace_trainer.distance_correlation)."""
    import torch

    a = a.reshape(-1, 1)
    b = b.reshape(-1, 1)
    A = (a - a.T).abs()
    B = (b - b.T).abs()
    A = A - A.mean(0, keepdim=True) - A.mean(1, keepdim=True) + A.mean()
    B = B - B.mean(0, keepdim=True) - B.mean(1, keepdim=True) + B.mean()
    dcov2 = (A * B).mean().clamp(min=0)
    denom = (A.pow(2).mean() * B.pow(2).mean()).sqrt().clamp(min=1e-12)
    return dcov2.sqrt() / denom.sqrt()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-interactions", required=True)
    ap.add_argument("--cf-artifacts", required=True, help="test-time artifacts (titles/pop/cat reused)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-path", default=None, help="defaults to config backbone_model")
    ap.add_argument("--config", default=None, help="optional CC-PACE yaml")
    ap.add_argument("--n-neg", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=0, help="0 -> config value")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--fold-b-frac", type=float, default=0.2)
    ap.add_argument("--max-panels", type=int, default=0, help="cap panels (smoke)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg: CCPaceConfig = load_cc_pace_config(args.config)
    model_path = args.model_path or cfg.backbone_model
    epochs = args.epochs or cfg.epochs
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # ---------------- data: train panels (prefix-only) ----------------
    train_rows = read_jsonl(args.train_interactions)
    seqs = build_user_prefix_sequences(train_rows)
    art = load_cf_artifacts(args.cf_artifacts)
    # item titles come from the task file recorded in the artifact provenance
    # (the artifact itself stores only neighbour titles); ids as fallback.
    titles: dict[str, str] = {}
    pop = {str(k): float(v) for k, v in art.get("item_popularity", {}).items()}
    cat = {str(k): str(v) for k, v in art.get("item_category", {}).items()}
    task_path = art.get("provenance", {}).get("task")
    if task_path and os.path.exists(task_path):
        titles = collect_titles(read_jsonl(task_path))
    vocab = sorted({i for s in seqs.values() for i in s})
    pop_w = [pop.get(i, 0.0) + 1.0 for i in vocab]

    # prefix-trained SASRec for train-panel CF (pseudo-positive excluded per user)
    prefix_rows = []
    for u, s in seqs.items():
        for j, iid in enumerate(s[:-1]):
            prefix_rows.append({"user_id": u, "item_id": iid, "timestamp": float(j)})
    print(f"prefix SASRec: {len(prefix_rows)} interactions", flush=True)
    sas_args = argparse.Namespace(
        seed=args.seed, max_seq_len=50, num_negatives=4, hidden_dim=64, num_layers=2,
        num_heads=2, dropout=0.2, batch_size=128, lr=1e-3, weight_decay=0.0, epochs=30,
    )
    sas_model, sas_idx, _, _ = train_sasrec(prefix_rows, sas_args)
    from llm4rec.trainers.sasrec import left_pad

    def cf_provider_for(user: str, cands: list[str]) -> PrecomputedCFProvider:
        seq_items = [sas_idx[i] for i in seqs[user][:-1] if i in sas_idx]
        scores: dict[str, float] = {}
        if seq_items:
            seq_t = torch.tensor([left_pad(seq_items, 50)], dtype=torch.long)
            in_vocab = [(j, sas_idx[c]) for j, c in enumerate(cands) if c in sas_idx]
            if in_vocab:
                with torch.no_grad():
                    raw = sas_model.score_items(
                        seq_t, torch.tensor([[ix for _, ix in in_vocab]], dtype=torch.long)
                    ).squeeze(0).numpy().astype(float)
                mu, sd = float(raw.mean()), float(raw.std())
                z = (raw - mu) / sd if sd > 1e-8 else raw * 0.0
                scores = {cands[j]: float(z[k]) for k, (j, _) in enumerate(in_vocab)}
        nbrs = {c: art["item_neighbors"].get(c, []) for c in cands}
        return PrecomputedCFProvider(
            scores={user: scores}, neighbors={user: nbrs},
            clusters={k: int(v) for k, v in art["item_clusters"].items()},
        )

    panels = []
    for u, s in sorted(seqs.items()):
        if len(s) < 2:
            continue
        positive = s[-1]
        prefix = s[:-1]
        banned = set(s)
        negs: list[str] = []
        while len(negs) < args.n_neg:
            pick = rng.choices(vocab, weights=pop_w, k=args.n_neg)
            negs.extend(i for i in pick if i not in banned and i not in negs)
        negs = negs[: args.n_neg]
        cands = [positive] + negs
        order = list(range(len(cands)))
        rng.shuffle(order)
        cands = [cands[i] for i in order]
        panels.append({
            "user_id": u,
            "cands": cands,
            "pos_idx": cands.index(positive),
            "prefix": prefix,
        })
    rng.shuffle(panels)
    if args.max_panels:
        panels = panels[: args.max_panels]
    n_b = max(1, int(len(panels) * args.fold_b_frac))
    fold_b, fold_a = panels[:n_b], panels[n_b:]
    print(f"panels: foldA={len(fold_a)} foldB={len(fold_b)}", flush=True)
    with open(os.path.join(args.out, "fold_assignment.json"), "w", encoding="utf-8") as fh:
        json.dump({"fold_a": [p["user_id"] for p in fold_a],
                   "fold_b": [p["user_id"] for p in fold_b]}, fh)

    # ---------------- model: Qwen3-8B + LoRA ----------------
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tok.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True
    )
    lora_cfg = LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate,
    )
    plan = build_training_plan(cfg)
    print(json.dumps(plan.__dict__, default=str), flush=True)

    # label tokenization invariants for the single-token surrogate
    n_panel = 1 + args.n_neg
    labels = [f"[{i:03d}]" for i in range(n_panel)]
    lab_ids = [tok(l, add_special_tokens=False).input_ids for l in labels]
    prefix_ids = lab_ids[0][:1]
    assert all(l[:1] == prefix_ids for l in lab_ids), "label '[' prefix must be shared"
    digit_ids = [l[1] for l in lab_ids]
    assert len(set(digit_ids)) == len(digit_ids), "digit tokens must be unique per label"

    total_steps = epochs * len(fold_a)
    step, accum = 0, 0
    log_rows = []
    t_hi, t_lo = plan.temperature_schedule
    for epoch in range(epochs):
        rng.shuffle(fold_a)
        for p in fold_a:
            user, cands, pos_idx = p["user_id"], p["cands"], p["pos_idx"]
            prefix_titles = [titles.get(i, i) for i in p["prefix"]]
            profile = profile_from_history(prefix_titles, domain="beauty")
            provider = cf_provider_for(user, cands)
            cand_dicts = [
                {"item_id": c, "category": cat.get(c, ""), "brand": "",
                 "keywords": titles.get(c, c), "attrs": "", "popularity": pop.get(c, 0.0)}
                for c in cands
            ]
            panel = schema_mod.render_panel(
                candidates=cand_dicts, history_titles=prefix_titles, profile=profile,
                cf_evidence=provider.evidence_tokens(user, cands) if cfg.use_cf_tokens else None,
                cfg=cfg, rng=random.Random(rng.randrange(1 << 30)),
            )
            prompt = schema_mod.build_prompt(panel, n_panel)
            enc = tok(prompt, return_tensors="pt", truncation=True,
                      max_length=cfg.max_context_tokens).input_ids
            inp = torch.cat([enc, torch.tensor([prefix_ids])], dim=1).to(args.device)
            logits = model(inp).logits[0, -1, :].float()
            logp = torch.log_softmax(logits, dim=-1)
            # presented position -> score; positive's PRESENTED position via original idx
            s = logp[torch.tensor(digit_ids[: len(cands)], device=logits.device)]
            pos_presented = panel.presentation_to_original.index(pos_idx)
            frac = step / max(1, total_steps - 1)
            temperature = t_hi + (t_lo - t_hi) * frac
            s_t = s / max(temperature, 1e-6)
            loss = torch.logsumexp(s_t, dim=0) - s_t[pos_presented]
            if cfg.use_dcor_penalty:
                lp = torch.tensor(
                    [np.log1p(max(pop.get(c, 0.0), 0.0)) for c in
                     (cands[i] for i in panel.presentation_to_original)],
                    device=s.device, dtype=s.dtype,
                )
                loss = loss + cfg.dcor_weight * torch_dcor(s, lp)
            (loss / args.grad_accum).backward()
            accum += 1
            if accum >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_(
                    [p_ for p_ in model.parameters() if p_.requires_grad], 1.0
                )
                opt.step()
                opt.zero_grad()
                accum = 0
            step += 1
            if step % 25 == 0:
                row = {"step": step, "epoch": epoch, "loss": round(float(loss.detach()), 4),
                       "temperature": round(temperature, 4)}
                log_rows.append(row)
                print(json.dumps(row), flush=True)

    model.save_pretrained(args.out)
    meta = {
        "plan": plan.__dict__,
        "surrogate": "single-digit-token PL (inference keeps full-label scoring)",
        "n_fold_a": len(fold_a), "n_fold_b": len(fold_b),
        "epochs": epochs, "steps": step, "seed": args.seed,
        "prefix_sasrec": "trained on per-user prefixes (pseudo-positive excluded)",
        "args": {k: v for k, v in vars(args).items()},
    }
    with open(os.path.join(args.out, "train_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    with open(os.path.join(args.out, "training_log.jsonl"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(r) for r in log_rows) + "\n")
    print(f"LoRA saved to {args.out}; steps={step}")


if __name__ == "__main__":
    main()
