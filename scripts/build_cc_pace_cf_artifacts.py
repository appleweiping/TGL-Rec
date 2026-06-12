#!/usr/bin/env python3
"""Build frozen CF artifacts for CC-PACE (the PrecomputedCFProvider payload).

Trains a small SASRec on the domain's TRAIN interactions only (the repo's own
model/dataset code), then emits one JSON artifact with everything the CC-PACE
ranker needs at inference time:

  user_scores[user][cand]   panel-z-scored SASRec affinity (in-vocab cands only)
  item_neighbors[item]      top-K nearest in-vocab item TITLES (CF-embedding cosine)
  item_clusters[item]       KMeans cluster id in CF-embedding space
  item_popularity[item]     raw TRAIN interaction count  (-> residualizer log_pop)
  item_category[item]       coarse title-lexicon facet   (-> residualizer facet_bucket)

CF stays a frozen conditioning sigma-field: CC-PACE never trains collaborative
parameters (docs/method_v2_decision_CC-PACE.md). Leakage discipline: the model
sees TRAIN interactions only; test positives are never in the training stream
(verified upstream: train_interactions.jsonl == pre-cutoff history).

Usage (server, CPU is fine at this scale):
  python scripts/build_cc_pace_cf_artifacts.py \
      --train-interactions data/domains/beauty/train_interactions.jsonl \
      --task <ranking_test.jsonl> --out outputs/cc_pace_beauty/cf_artifacts.json
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from llm4rec.methods.cc_pace.text_facets import coarse_category  # noqa: E402


def parse_listish(x):
    if isinstance(x, list):
        return x
    if not x:
        return []
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def collect_titles(task_rows: list[dict]) -> dict[str, str]:
    """item_id -> title, mined from every titled field in the task file."""
    titles: dict[str, str] = {}
    for d in task_rows:
        cids = [str(c) for c in parse_listish(d.get("candidate_item_ids"))]
        ctitles = [str(t) for t in parse_listish(d.get("candidate_titles"))]
        for cid, t in zip(cids, ctitles):
            if t and cid not in titles:
                titles[cid] = t
        hids = [str(h) for h in parse_listish(d.get("history_item_ids"))]
        htitles = [str(t) for t in parse_listish(d.get("history"))]
        for hid, t in zip(hids, htitles):
            if t and hid not in titles:
                titles[hid] = t
        pid, ptitle = d.get("positive_item_id"), d.get("positive_item_title")
        if pid and ptitle and str(pid) not in titles:
            titles[str(pid)] = str(ptitle)
    return titles


def train_sasrec(train_rows: list[dict], args) -> tuple:
    """Train the repo's SASRecModel on train interactions; return (model, item_to_idx)."""
    import torch
    from torch.utils.data import DataLoader

    from llm4rec.experiments.seeding import set_global_seed
    from llm4rec.models.sasrec import SASRecModel
    from llm4rec.trainers.sasrec import (
        SASRecSequenceDataset,
        _collate_sasrec,
        build_item_mappings,
    )

    set_global_seed(args.seed)
    vocab = sorted({str(r["item_id"]) for r in train_rows})
    item_to_idx, _ = build_item_mappings([{"item_id": i} for i in vocab])
    dataset = SASRecSequenceDataset(
        train_interactions=train_rows,
        item_to_idx=item_to_idx,
        max_seq_len=args.max_seq_len,
        num_negatives=args.num_negatives,
        seed=args.seed,
    )
    model = SASRecModel(
        num_items=len(item_to_idx),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=_collate_sasrec,
    )
    model.train()
    last_loss = None
    for epoch in range(args.epochs):
        epoch_losses = []
        for batch in loader:
            pos_scores = model.score_items(batch["input"], batch["positive"].unsqueeze(1))
            neg_scores = model.score_items(batch["input"], batch["negative"])
            loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        last_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch + 1}/{args.epochs} loss={last_loss:.4f}", flush=True)
    model.eval()
    return model, item_to_idx, last_loss, len(dataset)


def main(argv=None) -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-interactions", required=True)
    ap.add_argument("--task", required=True, help="same-candidate ranking jsonl (full schema)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", default="beauty")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--num-heads", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--max-seq-len", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--num-negatives", type=int, default=4)
    ap.add_argument("--clusters", type=int, default=32)
    ap.add_argument("--neighbors", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)

    import torch

    from llm4rec.trainers.sasrec import build_user_sequences, left_pad

    train_rows = read_jsonl(args.train_interactions)
    task_rows = read_jsonl(args.task)
    titles = collect_titles(task_rows)
    print(f"train interactions={len(train_rows)} task rows={len(task_rows)} titles={len(titles)}")

    model, item_to_idx, final_loss, n_examples = train_sasrec(train_rows, args)
    idx_to_item = {v: k for k, v in item_to_idx.items()}
    vocab_size = len(item_to_idx)

    # --- per-user panel scores (z-scored within each user's in-vocab candidates) ---
    sequences = build_user_sequences(train_rows)
    user_scores: dict[str, dict[str, float]] = {}
    skipped_users = 0
    for d in task_rows:
        user_id = str(d.get("user_id"))
        cands = [str(c) for c in parse_listish(d.get("candidate_item_ids"))]
        seq_items = [item_to_idx[i] for i in sequences.get(user_id, []) if i in item_to_idx]
        if not seq_items or not cands:
            skipped_users += 1
            continue
        seq = torch.tensor([left_pad(seq_items, args.max_seq_len)], dtype=torch.long)
        in_vocab = [(j, item_to_idx[c]) for j, c in enumerate(cands) if c in item_to_idx]
        if not in_vocab:
            skipped_users += 1
            continue
        idx_tensor = torch.tensor([[ix for _, ix in in_vocab]], dtype=torch.long)
        with torch.no_grad():
            raw = model.score_items(seq, idx_tensor).squeeze(0).numpy().astype(float)
        mu, sd = float(raw.mean()), float(raw.std())
        z = (raw - mu) / sd if sd > 1e-8 else raw * 0.0
        user_scores[user_id] = {cands[j]: round(float(z[k]), 4) for k, (j, _) in enumerate(in_vocab)}

    # --- item embedding geometry: clusters + nearest-neighbour titles ---
    with torch.no_grad():
        emb = model.item_embedding.weight[1:].numpy().astype(np.float64)  # row r -> idx r+1
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    unit = emb / np.clip(norms, 1e-9, None)

    from sklearn.cluster import KMeans

    k = min(args.clusters, max(2, vocab_size // 8))
    km = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit(unit)
    item_clusters = {idx_to_item[r + 1]: int(km.labels_[r]) for r in range(vocab_size)}

    sim = unit @ unit.T
    np.fill_diagonal(sim, -np.inf)
    nn_idx = np.argsort(-sim, axis=1)[:, : args.neighbors]
    item_neighbors: dict[str, list[str]] = {}
    for r in range(vocab_size):
        iid = idx_to_item[r + 1]
        nbr_titles = []
        for c in nn_idx[r]:
            nid = idx_to_item[int(c) + 1]
            nbr_titles.append(titles.get(nid, nid))
        item_neighbors[iid] = nbr_titles

    # --- train popularity + coarse category facets ---
    pop: dict[str, int] = {}
    for r in train_rows:
        iid = str(r["item_id"])
        pop[iid] = pop.get(iid, 0) + 1
    item_category = {iid: coarse_category(t, args.domain) for iid, t in titles.items()}

    artifact = {
        "domain": args.domain,
        "provenance": {
            "builder": "build_cc_pace_cf_artifacts.py",
            "cf_model": "sasrec",
            "train_interactions": os.path.abspath(args.train_interactions),
            "task": os.path.abspath(args.task),
            "n_train_rows": len(train_rows),
            "n_train_examples": n_examples,
            "vocab_size": vocab_size,
            "final_train_loss": final_loss,
            "scored_users": len(user_scores),
            "skipped_users": skipped_users,
            "score_normalization": "z-score within each user's in-vocab panel",
            "hyperparams": {
                "epochs": args.epochs, "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers, "num_heads": args.num_heads,
                "dropout": args.dropout, "max_seq_len": args.max_seq_len,
                "batch_size": args.batch_size, "lr": args.lr,
                "num_negatives": args.num_negatives, "clusters": k,
                "neighbors": args.neighbors, "seed": args.seed,
            },
        },
        "user_scores": user_scores,
        "item_neighbors": item_neighbors,
        "item_clusters": item_clusters,
        "item_popularity": pop,
        "item_category": item_category,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh)
    print(
        f"wrote {args.out}: users={len(user_scores)} vocab={vocab_size} "
        f"clusters={k} skipped={skipped_users}"
    )
    return args.out


if __name__ == "__main__":
    main()
