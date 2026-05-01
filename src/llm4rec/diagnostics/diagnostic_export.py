"""Phase 2B diagnostic export and summary assembly."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from llm4rec.data.text_fields import item_text
from llm4rec.diagnostics.statistics import target_rank
from llm4rec.experiments.config import load_yaml_config, resolve_path, save_resolved_config
from llm4rec.io.artifacts import read_jsonl, write_csv_rows, write_json
from llm4rec.rankers.bm25 import tokenize


def export_diagnostics(run_dir: str | Path, *, similarity_config: str | Path | None = None) -> dict[str, Any]:
    """Export similarity-vs-transition cases and numeric diagnostic summary."""

    root = Path(run_dir)
    processed = root / "artifacts" / "processed_dataset"
    if not processed.is_dir():
        raise FileNotFoundError(f"Missing processed dataset under run_dir: {processed}")
    config = load_yaml_config(similarity_config) if similarity_config else {}
    diagnostic = config.get("diagnostic", {})
    threshold = float(diagnostic.get("similarity_threshold", 0.2))
    top_k_pairs = int(diagnostic.get("top_k_pairs", 500))
    similarity = build_similarity_transition_cases(
        run_dir=root,
        similarity_threshold=threshold,
        top_k_pairs=top_k_pairs,
    )
    write_json(root / "similarity_vs_transition.json", similarity)
    write_csv_rows(root / "similarity_vs_transition.csv", similarity["cases"])
    write_json(root / "diagnostics" / "grouped_cases.json", similarity["grouped_cases"])
    summary = build_diagnostic_summary(root, similarity)
    write_json(root / "diagnostic_summary.json", summary)
    return summary


def run_phase2b_from_config(config_path: str | Path) -> Path:
    """Run perturbation, time-window, and export steps from one experiment config."""

    from llm4rec.diagnostics.perturbation_runner import run_perturbation_experiment
    from llm4rec.diagnostics.time_window_runner import run_time_window_experiment

    config = load_yaml_config(config_path)
    perturbation_config = config.get("perturbation_config")
    time_window_config = config.get("time_window_config")
    similarity_config = config.get("similarity_config")
    if not perturbation_config or not time_window_config:
        raise ValueError("Phase 2B config requires perturbation_config and time_window_config")
    run_dir = run_perturbation_experiment(resolve_path(perturbation_config))
    run_time_window_experiment(resolve_path(time_window_config))
    export_diagnostics(run_dir, similarity_config=resolve_path(similarity_config) if similarity_config else None)
    save_resolved_config(config, run_dir / "resolved_config.yaml")
    return run_dir


def build_similarity_transition_cases(
    *,
    run_dir: Path,
    similarity_threshold: float,
    top_k_pairs: int,
) -> dict[str, Any]:
    """Compute item-pair groups from text similarity and transition evidence."""

    items = read_jsonl(run_dir / "artifacts" / "processed_dataset" / "items.jsonl")
    item_by_id = {str(row["item_id"]): row for row in items}
    tokens_by_id = {item_id: set(tokenize(item_text(row))) for item_id, row in item_by_id.items()}
    transition_edges = _load_edges(run_dir / "diagnostics" / "transition_edges.jsonl")
    transition_map = {
        (str(edge["source_item"]), str(edge["target_item"])): edge
        for edge in transition_edges
    }
    window_maps = {
        label: _edge_score_map(run_dir / "diagnostics" / f"time_window_edges_{label}.jsonl")
        for label in ("1d", "7d", "30d")
    }
    group_counts = {
        "cross_category_transition": 0,
        "neither": 0,
        "same_category_transition": 0,
        "semantic_and_transition": 0,
        "semantic_only": 0,
        "transition_only": 0,
    }
    grouped_cases = {key: [] for key in group_counts}
    export_cases: list[dict[str, Any]] = []
    item_ids = sorted(item_by_id)
    for source in item_ids:
        for target in item_ids:
            if source == target:
                continue
            similarity = _jaccard(tokens_by_id[source], tokens_by_id[target])
            transition = transition_map.get((source, target))
            transition_count = float(transition.get("count", 0.0)) if transition else 0.0
            transition_score = float(transition.get("time_decayed_weight", transition_count)) if transition else 0.0
            same_category = item_by_id[source].get("category") == item_by_id[target].get("category")
            dominant_gap_bucket = _dominant_bucket(transition.get("bucket_counts", {}) if transition else {})
            row = {
                "dominant_gap_bucket": dominant_gap_bucket,
                "mean_time_gap": transition.get("mean_time_gap") if transition else None,
                "same_genre_or_category": same_category,
                "source_item": source,
                "source_title": item_by_id[source].get("title"),
                "target_item": target,
                "target_title": item_by_id[target].get("title"),
                "text_similarity": similarity,
                "time_window_score_1d": window_maps["1d"].get(tuple(sorted((source, target))), 0.0),
                "time_window_score_7d": window_maps["7d"].get(tuple(sorted((source, target))), 0.0),
                "time_window_score_30d": window_maps["30d"].get(tuple(sorted((source, target))), 0.0),
                "transition_count": transition_count,
                "transition_score": transition_score,
            }
            groups = _case_groups(row, similarity_threshold=similarity_threshold)
            for group in groups:
                group_counts[group] += 1
                if len(grouped_cases[group]) < 20:
                    grouped_cases[group].append(row)
            if transition_count > 0 or similarity >= similarity_threshold or any(row[f"time_window_score_{label}"] > 0 for label in ("1d", "7d", "30d")):
                row["primary_group"] = groups[0]
                export_cases.append(row)
    export_cases = sorted(
        export_cases,
        key=lambda row: (
            -float(row["transition_score"]),
            -float(row["time_window_score_7d"]),
            -float(row["text_similarity"]),
            str(row["source_item"]),
            str(row["target_item"]),
        ),
    )[:top_k_pairs]
    return {
        "cases": export_cases,
        "group_counts": group_counts,
        "grouped_cases": grouped_cases,
        "similarity_threshold": similarity_threshold,
    }


def build_diagnostic_summary(run_dir: Path, similarity: dict[str, Any]) -> dict[str, Any]:
    """Build numeric answers for the Phase 2B diagnostic checklist."""

    perturbation_rows = _read_csv(run_dir / "perturbation_results.csv")
    delta_rows = _read_csv(run_dir / "perturbation_deltas.csv")
    overlap_rows = _read_csv(run_dir / "prediction_overlap.csv")
    window_rows = _read_csv(run_dir / "time_window_graph_summary.csv")
    strongest = _strongest_delta(delta_rows)
    sensitivity = _baseline_sensitivity(delta_rows)
    overlap_values = [float(row.get("mean_prediction_overlap", 0.0)) for row in overlap_rows]
    best_window = _best_window(window_rows)
    transition_recovery = _transition_recovers_bm25(run_dir, best_window.get("window", ""))
    recent_beats = _recent_beats_original(perturbation_rows)
    group_counts = similarity["group_counts"]
    cross_count = int(group_counts.get("cross_category_transition", 0))
    transition_total = int(group_counts.get("cross_category_transition", 0)) + int(group_counts.get("same_category_transition", 0))
    return {
        "bm25_semantic_observation": {
            "bm25_order_delta_ndcg5_abs_mean": _mean_abs_delta(delta_rows, method="bm25"),
            "semantic_only_pairs": int(group_counts.get("semantic_only", 0)),
            "semantic_and_transition_pairs": int(group_counts.get("semantic_and_transition", 0)),
        },
        "cross_category_transition": {
            "count": cross_count,
            "ratio_among_transition_pairs": cross_count / float(transition_total or 1),
        },
        "evidence_for_phase3": {
            "perturbation_variants": len({row.get("perturbation") for row in perturbation_rows}),
            "prediction_overlap_min": min(overlap_values) if overlap_values else 0.0,
            "time_windows_evaluated": len(window_rows),
            "timestamp_bucketed_variant_present": any(row.get("perturbation") == "timestamp_bucketed_prompt_ready" for row in perturbation_rows),
        },
        "least_sequence_sensitive_baseline": sensitivity.get("least"),
        "most_sequence_sensitive_baseline": sensitivity.get("most"),
        "prediction_overlap_range": {
            "max": max(overlap_values) if overlap_values else 0.0,
            "min": min(overlap_values) if overlap_values else 0.0,
        },
        "recent_k_beats_full_history": recent_beats,
        "similarity_vs_transition_counts": group_counts,
        "strongest_perturbation_effect": strongest,
        "strongest_time_window_signal": best_window,
        "transition_graph_recovers_bm25_misses": transition_recovery,
    }


def _load_edges(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.is_file() else []


def _edge_score_map(path: Path) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    for edge in _load_edges(path):
        key = tuple(sorted((str(edge["source_item"]), str(edge["target_item"]))))
        scores[key] = scores.get(key, 0.0) + float(edge.get("time_decayed_weight", edge.get("weight", 0.0)))
    return scores


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / float(len(union))


def _dominant_bucket(bucket_counts: dict[str, Any]) -> str:
    if not bucket_counts:
        return ""
    return sorted(bucket_counts, key=lambda key: (-int(bucket_counts[key]), str(key)))[0]


def _case_groups(row: dict[str, Any], *, similarity_threshold: float) -> list[str]:
    semantic = float(row["text_similarity"]) >= similarity_threshold
    transition = float(row["transition_count"]) > 0
    groups: list[str] = []
    if semantic and transition:
        groups.append("semantic_and_transition")
    elif semantic:
        groups.append("semantic_only")
    elif transition:
        groups.append("transition_only")
    else:
        groups.append("neither")
    if transition and bool(row["same_genre_or_category"]):
        groups.append("same_category_transition")
    if transition and not bool(row["same_genre_or_category"]):
        groups.append("cross_category_transition")
    return groups


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _strongest_delta(rows: list[dict[str, str]]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    best_score = -1.0
    for row in rows:
        if row.get("perturbation") == "original":
            continue
        score = max(
            abs(float(row.get("delta_Recall@5_vs_original", 0.0) or 0.0)),
            abs(float(row.get("delta_NDCG@5_vs_original", 0.0) or 0.0)),
        )
        if score > best_score:
            best_score = score
            best = dict(row)
            best["absolute_effect"] = score
    return best


def _baseline_sensitivity(rows: list[dict[str, str]]) -> dict[str, Any]:
    scores: dict[str, list[float]] = {}
    for row in rows:
        if row.get("perturbation") == "original":
            continue
        method = str(row.get("method"))
        scores.setdefault(method, []).append(abs(float(row.get("delta_NDCG@5_vs_original", 0.0) or 0.0)))
    averages = {method: sum(values) / float(len(values) or 1) for method, values in scores.items()}
    if not averages:
        return {"least": None, "most": None, "scores": {}}
    most = max(averages, key=averages.get)
    least = min(averages, key=averages.get)
    return {
        "least": {"method": least, "mean_abs_delta_NDCG@5": averages[least]},
        "most": {"method": most, "mean_abs_delta_NDCG@5": averages[most]},
        "scores": averages,
    }


def _best_window(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {}
    metric = "Recall@5"
    best = max(rows, key=lambda row: float(row.get(metric, 0.0) or 0.0))
    return dict(best)


def _transition_recovers_bm25(run_dir: Path, best_window: str) -> dict[str, Any]:
    bm25_path = run_dir / "artifacts" / "predictions" / "bm25__original.jsonl"
    transition_path = run_dir / "artifacts" / "predictions" / f"transition_window_{best_window}.jsonl"
    if not bm25_path.is_file() or not transition_path.is_file():
        return {"bm25_missed_transition_hit_at5": 0, "compared": 0}
    bm25 = {
        (row["user_id"], row["target_item"]): row
        for row in read_jsonl(bm25_path)
    }
    recovered = 0
    compared = 0
    for row in read_jsonl(transition_path):
        key = (row["user_id"], row["target_item"])
        base = bm25.get(key)
        if base is None:
            continue
        compared += 1
        bm25_rank = target_rank(base["predicted_items"], base["target_item"])
        transition_rank = target_rank(row["predicted_items"], row["target_item"])
        if (bm25_rank is None or bm25_rank > 5) and transition_rank is not None and transition_rank <= 5:
            recovered += 1
    return {"bm25_missed_transition_hit_at5": recovered, "compared": compared}


def _recent_beats_original(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_key = {(row.get("method"), row.get("perturbation")): row for row in rows}
    output: list[dict[str, Any]] = []
    for method in sorted({row.get("method") for row in rows}):
        original = by_key.get((method, "original"))
        if original is None:
            continue
        for variant in ("recent_5", "recent_10"):
            row = by_key.get((method, variant))
            if row is None:
                continue
            output.append(
                {
                    "beats_on_NDCG@5": float(row.get("NDCG@5", 0.0) or 0.0) > float(original.get("NDCG@5", 0.0) or 0.0),
                    "beats_on_Recall@5": float(row.get("Recall@5", 0.0) or 0.0) > float(original.get("Recall@5", 0.0) or 0.0),
                    "method": method,
                    "variant": variant,
                }
            )
    return output


def _mean_abs_delta(rows: list[dict[str, str]], *, method: str) -> float:
    values = [
        abs(float(row.get("delta_NDCG@5_vs_original", 0.0) or 0.0))
        for row in rows
        if row.get("method") == method and row.get("perturbation") != "original"
    ]
    if not values:
        return 0.0
    return sum(values) / float(len(values))
