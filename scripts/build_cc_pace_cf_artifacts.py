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


def right_pad(values: list[int], max_seq_len: int) -> list[int]:
    """Right-pad with 0 so real items occupy the FIRST positions.

    SASRecModel.final_state reads the last *non-padding* item by gathering index
    ``ne(0).sum(dim=1) - 1``, which only points at the true last item when the
    sequence is RIGHT-padded. The repo's ``left_pad`` (real items at the end)
    makes that index land inside the left-padding region, whose state is masked to
    zero -> the SASRec final state is 0 -> every candidate scores 0 -> z-scoring a
    constant panel yields all-zero / NaN. Right-padding here aligns the builder
    with the model's contract and is the actual cause of the degenerate artifact.
    """
    clipped = list(values)[-int(max_seq_len):]
    return clipped + [0] * (int(max_seq_len) - len(clipped))


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
    # SASRecSequenceDataset left-pads each example's input (real items at the end),
    # but SASRecModel.final_state expects RIGHT-padding (see right_pad docstring). Left
    # padding makes the final state zero, so the model trains on a 0-signal target and
    # the loss is stuck at -log sigma(0) ~ 0.69. Re-pack every example with right_pad to
    # train against the real final-state contract (final_train_loss then drops < 0.05).
    from dataclasses import replace as _dc_replace

    dataset.examples = [
        _dc_replace(ex, input_indices=right_pad(
            [t for t in ex.input_indices if t != 0], args.max_seq_len))
        for ex in dataset.examples
    ]
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
    ap.add_argument("--epochs", type=int, default=50)
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

    from llm4rec.trainers.sasrec import build_user_sequences

    train_rows = read_jsonl(args.train_interactions)
    task_rows = read_jsonl(args.task)
    titles = collect_titles(task_rows)
    print(f"train interactions={len(train_rows)} task rows={len(task_rows)} titles={len(titles)}")

    model, item_to_idx, final_loss, n_examples = train_sasrec(train_rows, args)
    idx_to_item = {v: k for k, v in item_to_idx.items()}
    vocab_size = len(item_to_idx)

    # --- per-user panel scores (z-scored within each user's in-vocab candidates) ---
    # A user's CF panel is emitted ONLY if its raw SASRec affinities are non-degenerate
    # (finite + std above DEGENERATE_STD). Degenerate users are deliberately OMITTED from
    # user_scores: PrecomputedCFProvider.evidence_tokens/nuisance treat a missing user as
    # CF-ABSENT and degrade that user to text-only PACE -- which is correct, vs emitting an
    # all-zero / NaN panel that renders as semantic noise in the judge prompt.
    DEGENERATE_STD = 1e-3
    sequences = build_user_sequences(train_rows)
    user_scores: dict[str, dict[str, float]] = {}
    skipped_users = 0
    degenerate_users = 0
    for d in task_rows:
        user_id = str(d.get("user_id"))
        cands = [str(c) for c in parse_listish(d.get("candidate_item_ids"))]
        seq_items = [item_to_idx[i] for i in sequences.get(user_id, []) if i in item_to_idx]
        if not seq_items or not cands:
            skipped_users += 1
            continue
        seq = torch.tensor([right_pad(seq_items, args.max_seq_len)], dtype=torch.long)
        in_vocab = [(j, item_to_idx[c]) for j, c in enumerate(cands) if c in item_to_idx]
        if not in_vocab:
            skipped_users += 1
            continue
        idx_tensor = torch.tensor([[ix for _, ix in in_vocab]], dtype=torch.long)
        with torch.no_grad():
            raw = model.score_items(seq, idx_tensor).squeeze(0).numpy().astype(float)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        mu, sd = float(raw.mean()), float(raw.std())
        if not np.isfinite(sd) or sd <= DEGENERATE_STD:
            # degenerate panel -> mark CF ABSENT for this user (omit, do not emit zeros)
            degenerate_users += 1
            continue
        z = np.nan_to_num((raw - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
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
            "degenerate_users": degenerate_users,
            "score_normalization": "z-score within each user's in-vocab panel "
            "(degenerate/NaN panels omitted -> CF absent)",
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

    # --- validation guard: fail loudly on an undertrained / degenerate artifact ---
    # Emitted panels are recomputed here (post-z) to verify health independently of the
    # build path. n_panel_users is the number of users we attempted to emit a panel for
    # (emitted + degenerate); skipped users had no usable history/candidates at all.
    n_panel_users = len(user_scores) + degenerate_users
    n_allzero = 0
    n_nan = 0
    n_healthy = 0
    for panel in user_scores.values():
        vals = np.asarray(list(panel.values()), dtype=float)
        if vals.size == 0 or np.isnan(vals).any():
            n_nan += 1
            continue
        if np.allclose(vals, 0.0):
            n_allzero += 1
        if float(np.std(vals)) > 1e-3:
            n_healthy += 1
    # degenerate users are intentionally absent, but count them against the panel budget
    degenerate_frac = (degenerate_users + n_allzero + n_nan) / max(n_panel_users, 1)
    healthy_frac = n_healthy / max(n_panel_users, 1)
    print(
        "[validate] final_train_loss={loss}  panel_users={pu}  emitted={em}  "
        "absent_degenerate={dg}  healthy(std>1e-3)={hh} ({hp:.1%})  "
        "allzero={az}  nan={nn}  degenerate_frac={df:.1%}".format(
            loss=final_loss, pu=n_panel_users, em=len(user_scores),
            dg=degenerate_users, hh=n_healthy, hp=healthy_frac,
            az=n_allzero, nn=n_nan, df=degenerate_frac,
        ),
        flush=True,
    )
    problems = []
    if final_loss is not None and final_loss >= 0.4:
        problems.append(f"final_train_loss={final_loss:.4f} >= 0.4 (SASRec undertrained)")
    if n_nan > 0:
        problems.append(f"{n_nan} emitted panels still contain NaN")
    if degenerate_frac > 0.20:
        problems.append(f"{degenerate_frac:.1%} of panels degenerate (>20% threshold)")
    if problems:
        print("[validate] FAILED:\n  - " + "\n  - ".join(problems), file=sys.stderr, flush=True)
        sys.exit(2)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh)
    print(
        f"wrote {args.out}: users={len(user_scores)} vocab={vocab_size} "
        f"clusters={k} skipped={skipped_users} degenerate={degenerate_users}"
    )
    return args.out


if __name__ == "__main__":
    main()
