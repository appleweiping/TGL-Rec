"""Time-window graph diagnostics for MovieLens-style data."""

from __future__ import annotations

from pathlib import Path
from statistics import fmean
from typing import Any

from llm4rec.data.preprocess import preprocess_from_config
from llm4rec.diagnostics.perturbation_runner import (
    _candidate_to_example,
    _prepare_run,
    _resolve_diagnostic_config,
)
from llm4rec.evaluation.diagnostic_evaluator import evaluate_diagnostic_predictions
from llm4rec.experiments.config import resolve_path
from llm4rec.graph.movie_transition_graph import build_movie_transition_edges
from llm4rec.graph.time_window_graph import build_time_window_edges
from llm4rec.io.artifacts import ensure_dir, read_jsonl, write_csv_rows, write_json, write_jsonl

WINDOW_SECONDS = {
    "1h": 60 * 60,
    "1d": 24 * 60 * 60,
    "7d": 7 * 24 * 60 * 60,
    "30d": 30 * 24 * 60 * 60,
    "full": 10**15,
}


def run_time_window_experiment(config_path: str | Path) -> Path:
    """Build time-window graphs and evaluate simple transition-strength rankers."""

    config = _resolve_diagnostic_config(config_path)
    run_dir, logger = _prepare_run(config)
    processed_dir = _prepare_processed_dataset(config, run_dir, logger)
    dataset = config["dataset"]
    evaluation = config.get("evaluation", {})
    diagnostic = config.get("diagnostic", {})
    windows = [str(window) for window in diagnostic.get("windows", ["1h", "1d", "7d", "30d", "full"])]
    half_life = float(diagnostic.get("half_life_seconds", 604800))
    max_history_items = int(diagnostic.get("max_history_items", 10))
    ks = tuple(int(k) for k in evaluation.get("ks", [1, 3, 5]))
    candidate_protocol = str(evaluation.get("candidate_protocol", dataset.get("candidate_protocol", "full_catalog")))

    train_rows = read_jsonl(processed_dir / "train.jsonl")
    all_interactions = read_jsonl(processed_dir / "interactions.jsonl")
    item_rows = read_jsonl(processed_dir / "items.jsonl")
    candidate_rows = read_jsonl(processed_dir / "candidates.jsonl")
    item_catalog = {str(row["item_id"]) for row in item_rows}
    diagnostics_dir = ensure_dir(run_dir / "diagnostics")
    graphs_dir = ensure_dir(run_dir / "artifacts" / "graphs")
    predictions_dir = ensure_dir(run_dir / "artifacts" / "predictions")

    transition_edges = build_movie_transition_edges(train_rows, half_life_seconds=half_life)
    write_jsonl(diagnostics_dir / "transition_edges.jsonl", transition_edges)
    write_jsonl(graphs_dir / "directed_transition_edges.jsonl", transition_edges)
    examples = [_candidate_to_example(row, all_interactions) for row in candidate_rows if row["split"] == "test"]
    summary_rows: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    for window in windows:
        seconds = WINDOW_SECONDS[window]
        edges = build_time_window_edges(
            train_rows,
            window_seconds=seconds,
            directed=False,
            weight_mode="time_decay",
            half_life_seconds=half_life,
        )
        suffix = "full" if window == "full" else window
        write_jsonl(diagnostics_dir / f"time_window_edges_{suffix}.jsonl", edges)
        write_jsonl(graphs_dir / f"time_window_edges_{suffix}.jsonl", edges)
        predictions = _transition_window_predictions(
            edges,
            examples=examples,
            window=window,
            max_history_items=max_history_items,
        )
        write_jsonl(predictions_dir / f"transition_window_{window}.jsonl", predictions)
        all_predictions.extend(predictions)
        metric_rows = evaluate_diagnostic_predictions(
            prediction_rows=predictions,
            item_catalog=item_catalog,
            ks=ks,
            candidate_protocol=candidate_protocol,
        )
        metrics = metric_rows[0] if metric_rows else {}
        summary_rows.append(
            {
                **_summarize_window_edges(edges, window=window),
                **{key: value for key, value in metrics.items() if key not in {"method", "perturbation"}},
                "method": f"transition_window_{window}",
            }
        )
    write_jsonl(predictions_dir / "transition_window_all.jsonl", all_predictions)
    write_csv_rows(run_dir / "time_window_graph_summary.csv", summary_rows)
    write_json(run_dir / "artifacts" / "graphs" / "time_window_graph_summary.json", {"windows": summary_rows})
    logger.info("time-window diagnostics completed")
    return run_dir


def _prepare_processed_dataset(config: dict[str, Any], run_dir: Path, logger: Any) -> Path:
    dataset = dict(config["dataset"])
    target = run_dir / "artifacts" / "processed_dataset"
    if target.is_dir() and (target / "interactions.jsonl").is_file():
        return target
    if bool(dataset.get("preprocess", True)):
        dataset["output_dir"] = str(target)
        logger.info("running MovieLens-style preprocessing for time windows")
        return preprocess_from_config({"dataset": dataset}).output_dir
    return resolve_path(dataset["output_dir"])


def _transition_window_predictions(
    edges: list[dict[str, Any]],
    *,
    examples: list[Any],
    window: str,
    max_history_items: int,
) -> list[dict[str, Any]]:
    scores_by_item: dict[str, dict[str, float]] = {}
    for edge in edges:
        source = str(edge["source_item"])
        target = str(edge["target_item"])
        weight = float(edge.get("time_decayed_weight", edge.get("weight", 0.0)))
        scores_by_item.setdefault(source, {})[target] = scores_by_item.setdefault(source, {}).get(target, 0.0) + weight
        scores_by_item.setdefault(target, {})[source] = scores_by_item.setdefault(target, {}).get(source, 0.0) + weight
    predictions: list[dict[str, Any]] = []
    method = f"transition_window_{window}"
    for example in examples:
        scores: dict[str, float] = {}
        history = example.history[-max_history_items:]
        for candidate in example.candidate_items:
            candidate_id = str(candidate)
            scores[candidate_id] = sum(
                scores_by_item.get(str(source), {}).get(candidate_id, 0.0)
                for source in history
            )
        ranked = sorted(example.candidate_items, key=lambda item_id: (-scores.get(str(item_id), 0.0), str(item_id)))
        predictions.append(
            {
                "candidate_items": example.candidate_items,
                "domain": example.domain,
                "metadata": {
                    "max_history_items": max_history_items,
                    "perturbation": window,
                    "phase": "phase2b_time_window",
                    "window": window,
                },
                "method": method,
                "predicted_items": [str(item) for item in ranked],
                "raw_output": None,
                "scores": [float(scores[str(item)]) for item in ranked],
                "target_item": example.target_item,
                "user_id": example.user_id,
            }
        )
    return predictions


def _summarize_window_edges(edges: list[dict[str, Any]], *, window: str) -> dict[str, Any]:
    gap_values = [float(edge["mean_time_gap"]) for edge in edges if edge.get("mean_time_gap") is not None]
    bucket_counts: dict[str, int] = {}
    for edge in edges:
        for bucket, count in dict(edge.get("bucket_counts", {})).items():
            bucket_counts[str(bucket)] = bucket_counts.get(str(bucket), 0) + int(count)
    dominant = sorted(bucket_counts, key=lambda key: (-bucket_counts[key], key))[0] if bucket_counts else ""
    return {
        "dominant_gap_bucket": dominant,
        "edge_count": len(edges),
        "mean_time_gap": fmean(gap_values) if gap_values else 0.0,
        "time_decayed_weight_sum": sum(float(edge.get("time_decayed_weight", 0.0)) for edge in edges),
        "user_count": sum(int(edge.get("user_count", 0)) for edge in edges),
        "window": window,
    }
