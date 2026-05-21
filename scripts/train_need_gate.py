"""Train the TGL-Rec need-gate and prepare LoRA training data.

Usage:
    python scripts/train_need_gate.py \
        --config configs/experiments/tglrec_gate_train.yaml \
        --output-dir outputs/gate_training/

This script:
1. Loads train interactions and builds TDIG
2. Computes need-state for each user
3. Retrieves evidence for all (user, candidate) pairs
4. Trains the lightweight need-gate (logistic, ~26 params)
5. Exports gate weights and training diagnostics
6. Prepares LoRA training data with evidence text
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm4rec.evidence.need_gate import GateConfig, LearnedNeedGate
from llm4rec.evidence.need_state import NeedStateEncoder
from llm4rec.evidence.reportable_scorer import ReportableScorer
from llm4rec.evidence.retriever import TemporalEvidenceRetriever
from llm4rec.evidence.temporal_graph import build_temporal_graph_artifacts
from llm4rec.experiments.config import load_yaml_config, resolve_experiment_config
from llm4rec.io.artifacts import ensure_dir, read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TGL-Rec need-gate")
    parser.add_argument("--config", required=True, help="Experiment config YAML")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-users", type=int, default=None, help="Limit users for debugging")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    print(f"[gate-train] Loading config: {args.config}")
    config = resolve_experiment_config(args.config)

    train_path = Path(config["dataset"]["train_interactions"])
    items_path = Path(config["dataset"]["item_records"])
    candidates_dir = Path(config["dataset"]["candidates_dir"])

    print(f"[gate-train] Loading train interactions from {train_path}")
    train_interactions = list(read_jsonl(train_path))
    item_records = list(read_jsonl(items_path))
    item_categories = {str(r["item_id"]): str(r.get("category", "unknown")) for r in item_records}

    print(f"[gate-train] Building TDIG from {len(train_interactions)} interactions")
    t0 = time.time()
    graph = build_temporal_graph_artifacts(
        train_interactions=train_interactions,
        output_dir=output_dir / "graph_cache",
        window_seconds=float(config.get("time_window", {}).get("window_seconds", 86400)),
        candidate_protocol=config.get("evaluation", {}).get("candidate_protocol", "same_candidate"),
    )
    print(f"[gate-train] TDIG built in {time.time() - t0:.1f}s")

    transition_edges = list(graph["transition_edges"])
    time_window_edges = list(graph["time_window_edges"])
    transition_index = _build_transition_index(transition_edges)

    need_encoder = NeedStateEncoder(
        recent_window=5,
        tau_max=30 * 86400,
        transition_index=transition_index,
        item_categories=item_categories,
    )

    retriever = TemporalEvidenceRetriever(
        transition_edges=transition_edges,
        time_window_edges=time_window_edges,
        item_records=item_records,
        config=dict(config.get("retrieval", {})),
        transition_artifact=str(graph["transition_path"]),
        time_window_artifact=str(graph["time_window_path"]),
        candidate_protocol="same_candidate",
        constructed_from="train_only",
    )

    print("[gate-train] Loading candidate sets and ground truth")
    user_histories, user_timestamps, candidate_sets, ground_truth = _load_train_eval_data(
        train_interactions, candidates_dir, max_users=args.max_users
    )

    print(f"[gate-train] Generating gate training data for {len(user_histories)} users")
    scorer = ReportableScorer(
        gate=LearnedNeedGate(GateConfig.from_dict(config.get("gate", {}))),
        need_state_encoder=need_encoder,
    )

    all_examples: list[dict] = []
    for user_id in user_histories:
        history = user_histories[user_id]
        timestamps = user_timestamps.get(user_id)
        candidates = candidate_sets.get(user_id, [])
        gt = ground_truth.get(user_id)
        if not candidates or not gt:
            continue

        need_state = scorer.set_user_context(
            history=history,
            timestamps=timestamps,
            prediction_timestamp=timestamps[-1] + 1 if timestamps else None,
        )
        need_vec = need_state.to_vector()

        retrieval = retriever.retrieve(
            user_id=user_id,
            history=history,
            candidate_items=candidates,
            prediction_timestamp=timestamps[-1] + 1 if timestamps else None,
        )

        gt_rows = [r for r in retrieval.evidence if str(r.target_item) == gt]
        gt_features = ReportableScorer._extract_evidence_features(gt_rows)
        gt_temporal = sum(gt_features[:6])

        if gt_temporal > 0:
            all_examples.append({
                "need_state": need_vec,
                "evidence_features": gt_features,
                "label": 1,
                "weight": min(1.0, gt_temporal),
            })

        for cand in random.sample(candidates, min(5, len(candidates))):
            if cand == gt:
                continue
            cand_rows = [r for r in retrieval.evidence if str(r.target_item) == cand]
            cand_features = ReportableScorer._extract_evidence_features(cand_rows)
            cand_semantic = cand_features[6]
            cand_temporal = sum(cand_features[:6])
            if cand_semantic > 0.2 and cand_temporal < cand_semantic * 0.5:
                all_examples.append({
                    "need_state": need_vec,
                    "evidence_features": cand_features,
                    "label": 0,
                    "weight": min(1.0, cand_semantic),
                })

    print(f"[gate-train] Generated {len(all_examples)} examples "
          f"(pos={sum(1 for e in all_examples if e['label']==1)}, "
          f"neg={sum(1 for e in all_examples if e['label']==0)})")

    random.shuffle(all_examples)
    split = int(0.8 * len(all_examples))
    train_ex = all_examples[:split]
    valid_ex = all_examples[split:]

    print("[gate-train] Training need-gate...")
    gate = LearnedNeedGate(GateConfig.from_dict(config.get("gate", {})))
    metrics = gate.train(train_ex, valid_ex)
    print(f"[gate-train] Training complete: {metrics}")

    gate.save(output_dir / "need_gate_weights.json")
    write_json(output_dir / "gate_train_metrics.json", metrics)
    write_json(output_dir / "gate_train_config.json", {
        "config_path": str(args.config),
        "seed": args.seed,
        "n_train": len(train_ex),
        "n_valid": len(valid_ex),
        "n_users": len(user_histories),
    })

    print(f"[gate-train] Gate saved to {output_dir / 'need_gate_weights.json'}")
    print("[gate-train] Done.")


def _build_transition_index(edges: list[dict]) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = {}
    for edge in edges:
        src = str(edge.get("source", ""))
        if src not in index:
            index[src] = []
        index[src].append(edge)
    return index


def _load_train_eval_data(
    train_interactions: list[dict],
    candidates_dir: Path,
    max_users: int | None = None,
) -> tuple[dict, dict, dict, dict]:
    user_seqs: dict[str, list[tuple[str, float]]] = {}
    for row in train_interactions:
        uid = str(row["user_id"])
        iid = str(row["item_id"])
        ts = float(row.get("timestamp", 0))
        if uid not in user_seqs:
            user_seqs[uid] = []
        user_seqs[uid].append((iid, ts))

    for uid in user_seqs:
        user_seqs[uid].sort(key=lambda x: x[1])

    user_histories: dict[str, list[str]] = {}
    user_timestamps: dict[str, list[float]] = {}
    ground_truth: dict[str, str] = {}

    for uid, seq in user_seqs.items():
        if len(seq) < 3:
            continue
        user_histories[uid] = [s[0] for s in seq[:-1]]
        user_timestamps[uid] = [s[1] for s in seq[:-1]]
        ground_truth[uid] = seq[-1][0]

    if max_users and len(user_histories) > max_users:
        keys = random.sample(list(user_histories.keys()), max_users)
        user_histories = {k: user_histories[k] for k in keys}
        user_timestamps = {k: user_timestamps[k] for k in keys}
        ground_truth = {k: ground_truth[k] for k in keys}

    candidate_sets: dict[str, list[str]] = {}
    ranking_file = candidates_dir / "ranking_valid.jsonl"
    if ranking_file.exists():
        for row in read_jsonl(ranking_file):
            uid = str(row.get("user_id", ""))
            cands = [str(c) for c in row.get("candidate_items", [])]
            if uid and cands:
                candidate_sets[uid] = cands
    else:
        all_items = list({str(r["item_id"]) for r in train_interactions})
        for uid in user_histories:
            candidate_sets[uid] = random.sample(all_items, min(101, len(all_items)))

    return user_histories, user_timestamps, candidate_sets, ground_truth


if __name__ == "__main__":
    main()
