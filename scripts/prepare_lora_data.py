"""Prepare LoRA training data for TGL-Rec Stage 2 reranking.

Usage:
    python scripts/prepare_lora_data.py \
        --config configs/experiments/tglrec_lora_train.yaml \
        --gate-weights outputs/gate_training/need_gate_weights.json \
        --output-dir outputs/lora_data/

This script:
1. Loads trained gate weights
2. Runs Stage 1 scoring on all train users
3. Selects top-K candidates per user
4. Translates evidence to natural language
5. Formats as instruction-tuning data for Qwen3-8B LoRA
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm4rec.evidence.need_gate import LearnedNeedGate
from llm4rec.evidence.need_state import NeedStateEncoder
from llm4rec.evidence.reportable_scorer import ReportableScorer
from llm4rec.evidence.retriever import TemporalEvidenceRetriever
from llm4rec.evidence.temporal_graph import build_temporal_graph_artifacts
from llm4rec.evidence.translator import GraphToTextTranslator
from llm4rec.experiments.config import resolve_experiment_config
from llm4rec.io.artifacts import ensure_dir, read_jsonl, write_json, write_jsonl


SYSTEM_PROMPT = (
    "You are a recommendation assistant. Given a user's interaction history "
    "and temporal evidence about candidate items, rank the candidates from "
    "most to least likely to be the user's next interaction. Consider both "
    "the user's preferences and temporal transition patterns."
)


def format_history_text(history: list[str], item_records: dict[str, dict], max_items: int = 10) -> str:
    recent = history[-max_items:]
    lines = []
    for i, item_id in enumerate(recent, 1):
        rec = item_records.get(item_id, {})
        title = rec.get("title", rec.get("name", item_id))
        cat = rec.get("category", "")
        line = f"{i}. {title}"
        if cat:
            line += f" [{cat}]"
        lines.append(line)
    return "Recent interactions (oldest to newest):\n" + "\n".join(lines)


def format_evidence_text(
    evidence_by_candidate: dict[str, list],
    item_records: dict[str, dict],
    translator: GraphToTextTranslator,
    top_k: int = 5,
) -> str:
    lines = []
    for item_id, evidence_rows in list(evidence_by_candidate.items())[:top_k]:
        rec = item_records.get(item_id, {})
        title = rec.get("title", rec.get("name", item_id))
        if evidence_rows:
            text = translator.translate_evidence_list(evidence_rows)
            lines.append(f"- {title}: {text}")
        else:
            lines.append(f"- {title}: No temporal evidence")
    if lines:
        return "Temporal evidence for top candidates:\n" + "\n".join(lines)
    return ""


def format_candidates_text(candidates: list[str], item_records: dict[str, dict]) -> str:
    lines = []
    for i, item_id in enumerate(candidates, 1):
        rec = item_records.get(item_id, {})
        title = rec.get("title", rec.get("name", item_id))
        lines.append(f"{i}. {title}")
    return "Candidates to rank:\n" + "\n".join(lines)


def format_target_ranking(
    ground_truth: str,
    candidates: list[str],
    item_records: dict[str, dict],
) -> str:
    ordered = [ground_truth] + [c for c in candidates if c != ground_truth]
    lines = []
    for i, item_id in enumerate(ordered, 1):
        rec = item_records.get(item_id, {})
        title = rec.get("title", rec.get("name", item_id))
        lines.append(f"{i}. {title}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LoRA training data")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gate-weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    config = resolve_experiment_config(args.config)

    gate = LearnedNeedGate()
    gate.load(Path(args.gate_weights))
    print(f"[lora-data] Loaded gate weights (trained={gate.weights.trained})")

    train_path = Path(config["dataset"]["train_interactions"])
    items_path = Path(config["dataset"]["item_records"])
    candidates_dir = Path(config["dataset"]["candidates_dir"])

    train_interactions = list(read_jsonl(train_path))
    item_records_list = list(read_jsonl(items_path))
    item_records = {str(r["item_id"]): r for r in item_records_list}
    item_categories = {k: str(v.get("category", "unknown")) for k, v in item_records.items()}

    graph = build_temporal_graph_artifacts(
        train_interactions=train_interactions,
        output_dir=output_dir / "graph_cache",
        window_seconds=float(config.get("time_window", {}).get("window_seconds", 86400)),
        candidate_protocol="same_candidate",
    )

    transition_index = {}
    for edge in graph["transition_edges"]:
        src = str(edge.get("source", ""))
        if src not in transition_index:
            transition_index[src] = []
        transition_index[src].append(edge)

    need_encoder = NeedStateEncoder(
        recent_window=5,
        tau_max=30 * 86400,
        transition_index=transition_index,
        item_categories=item_categories,
    )

    retriever = TemporalEvidenceRetriever(
        transition_edges=list(graph["transition_edges"]),
        time_window_edges=list(graph["time_window_edges"]),
        item_records=item_records_list,
        config=dict(config.get("retrieval", {})),
        transition_artifact=str(graph["transition_path"]),
        time_window_artifact=str(graph["time_window_path"]),
        candidate_protocol="same_candidate",
        constructed_from="train_only",
    )

    scorer = ReportableScorer(gate=gate, need_state_encoder=need_encoder)
    translator = GraphToTextTranslator(
        str(config.get("translator", {}).get("mode", "prompt_ready_json"))
    )

    user_seqs: dict[str, list[tuple[str, float]]] = {}
    for row in train_interactions:
        uid = str(row["user_id"])
        if uid not in user_seqs:
            user_seqs[uid] = []
        user_seqs[uid].append((str(row["item_id"]), float(row.get("timestamp", 0))))
    for uid in user_seqs:
        user_seqs[uid].sort(key=lambda x: x[1])

    ranking_file = candidates_dir / "ranking_valid.jsonl"
    candidate_sets: dict[str, list[str]] = {}
    ground_truth_map: dict[str, str] = {}
    if ranking_file.exists():
        for row in read_jsonl(ranking_file):
            uid = str(row.get("user_id", ""))
            cands = [str(c) for c in row.get("candidate_items", [])]
            gt = str(row.get("target_item", ""))
            if uid and cands:
                candidate_sets[uid] = cands
                if gt:
                    ground_truth_map[uid] = gt

    users = list(candidate_sets.keys())
    if args.max_users and len(users) > args.max_users:
        users = random.sample(users, args.max_users)

    print(f"[lora-data] Preparing data for {len(users)} users, top-K={args.top_k}")
    lora_examples = []
    skipped = 0

    for i, uid in enumerate(users):
        if i % 500 == 0:
            print(f"[lora-data] Processing user {i}/{len(users)}")

        seq = user_seqs.get(uid, [])
        if len(seq) < 3:
            skipped += 1
            continue

        history = [s[0] for s in seq[:-1]]
        timestamps = [s[1] for s in seq[:-1]]
        candidates = candidate_sets[uid]
        gt = ground_truth_map.get(uid, seq[-1][0])

        retrieval = retriever.retrieve(
            user_id=uid,
            history=history,
            candidate_items=candidates,
            prediction_timestamp=timestamps[-1] + 1,
        )

        scorer.set_user_context(
            history=history,
            timestamps=timestamps,
            prediction_timestamp=timestamps[-1] + 1,
        )
        scores = scorer.score_candidates(retrieval.evidence, candidates, history=history)

        sorted_cands = sorted(scores.items(), key=lambda x: x[1].total_score, reverse=True)
        top_k_items = [item_id for item_id, _ in sorted_cands[: args.top_k]]

        if gt not in top_k_items:
            top_k_items = top_k_items[: args.top_k - 1] + [gt]

        evidence_by_cand = {}
        for item_id in top_k_items:
            evidence_by_cand[item_id] = [
                r for r in retrieval.evidence if str(r.target_item) == item_id
            ]

        history_text = format_history_text(history, item_records)
        evidence_text = format_evidence_text(evidence_by_cand, item_records, translator, top_k=5)
        candidates_text = format_candidates_text(top_k_items, item_records)
        target_text = format_target_ranking(gt, top_k_items, item_records)

        user_prompt = f"{history_text}\n\n{evidence_text}\n\n{candidates_text}\n\nRank these candidates:"
        lora_examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": target_text},
            ],
            "metadata": {
                "user_id": uid,
                "ground_truth": gt,
                "top_k_size": len(top_k_items),
                "gt_in_top_k": gt in top_k_items[:args.top_k],
            },
        })

    print(f"[lora-data] Generated {len(lora_examples)} examples (skipped {skipped})")

    random.shuffle(lora_examples)
    split = int(0.9 * len(lora_examples))
    train_data = lora_examples[:split]
    valid_data = lora_examples[split:]

    write_jsonl(output_dir / "train.jsonl", train_data)
    write_jsonl(output_dir / "valid.jsonl", valid_data)
    write_json(output_dir / "data_stats.json", {
        "total_examples": len(lora_examples),
        "train_examples": len(train_data),
        "valid_examples": len(valid_data),
        "top_k": args.top_k,
        "seed": args.seed,
    })

    print(f"[lora-data] Saved to {output_dir}")
    print(f"[lora-data] Train: {len(train_data)}, Valid: {len(valid_data)}")


if __name__ == "__main__":
    main()
