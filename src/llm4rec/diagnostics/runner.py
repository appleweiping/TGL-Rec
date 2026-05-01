"""Config-driven diagnostic artifact builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.data.preprocess import preprocess_from_config
from llm4rec.data.time_features import build_time_feature_rows
from llm4rec.diagnostics.sequence_perturbation import build_sequence_perturbation_artifact
from llm4rec.diagnostics.similarity_vs_transition import build_similarity_vs_transition_artifact
from llm4rec.experiments.config import load_yaml_config, resolve_path, save_resolved_config
from llm4rec.experiments.logging import RunLogger
from llm4rec.experiments.seeding import set_global_seed
from llm4rec.graph.time_window_graph import build_time_window_edges
from llm4rec.graph.transition_graph import build_transition_edges
from llm4rec.io.artifacts import ensure_dir, read_jsonl, write_json, write_jsonl
from llm4rec.metrics.transition import mean_transition_count, transition_coverage
from llm4rec.utils.env import collect_environment


def build_graph_artifacts(config_path: str | Path) -> Path:
    """Build transition and time-window graph artifacts from a diagnostics config."""

    config = _resolve_diagnostics_config(config_path)
    run_dir, diagnostics_dir, logger = _prepare_run(config)
    processed_dir = _prepare_processed_dataset(config, run_dir, logger)
    interactions = read_jsonl(processed_dir / "interactions.jsonl")
    items = read_jsonl(processed_dir / "items.jsonl")
    diagnostic = config.get("diagnostic", {})
    window_seconds = float(diagnostic.get("window_seconds", 604800))
    half_life = float(diagnostic.get("half_life_seconds", window_seconds))
    transition_edges = build_transition_edges(interactions)
    time_window_edges = build_time_window_edges(
        interactions,
        window_seconds=window_seconds,
        directed=bool(diagnostic.get("directed", False)),
        weight_mode=str(diagnostic.get("weight_mode", "time_decay")),
        half_life_seconds=half_life,
    )
    write_jsonl(diagnostics_dir / "transition_edges.jsonl", transition_edges)
    write_jsonl(diagnostics_dir / "time_window_edges.jsonl", time_window_edges)
    write_jsonl(diagnostics_dir / "time_features.jsonl", build_time_feature_rows(interactions))
    summary = _diagnostics_summary(transition_edges, time_window_edges, items)
    write_json(diagnostics_dir / "diagnostics_summary.json", summary)
    logger.info("graph diagnostics completed")
    return run_dir


def run_diagnostics(config_path: str | Path) -> Path:
    """Run all Phase 2A diagnostic artifact builders."""

    config = _resolve_diagnostics_config(config_path)
    run_dir, diagnostics_dir, logger = _prepare_run(config)
    processed_dir = _prepare_processed_dataset(config, run_dir, logger)
    interactions = read_jsonl(processed_dir / "interactions.jsonl")
    items = read_jsonl(processed_dir / "items.jsonl")
    diagnostic = config.get("diagnostic", {})
    seed = int(config.get("experiment", {}).get("seed", diagnostic.get("seed", 2026)))
    window_seconds = float(diagnostic.get("window_seconds", 604800))
    transition_edges = build_transition_edges(interactions)
    time_window_edges = build_time_window_edges(
        interactions,
        window_seconds=window_seconds,
        directed=bool(diagnostic.get("directed", False)),
        weight_mode=str(diagnostic.get("weight_mode", "time_decay")),
        half_life_seconds=float(diagnostic.get("half_life_seconds", window_seconds)),
    )
    sequence_artifact = build_sequence_perturbation_artifact(
        interactions,
        seed=seed,
        recent_k=int(diagnostic.get("recent_k", 2)),
    )
    similarity_artifact = build_similarity_vs_transition_artifact(
        interactions,
        items,
        window_seconds=window_seconds,
        top_k=int(diagnostic.get("top_k_pairs", 50)),
    )
    write_json(diagnostics_dir / "sequence_perturbation.json", sequence_artifact)
    write_jsonl(diagnostics_dir / "time_features.jsonl", build_time_feature_rows(interactions))
    write_jsonl(diagnostics_dir / "transition_edges.jsonl", transition_edges)
    write_jsonl(diagnostics_dir / "time_window_edges.jsonl", time_window_edges)
    write_json(diagnostics_dir / "similarity_vs_transition.json", similarity_artifact)
    summary = _diagnostics_summary(transition_edges, time_window_edges, items)
    summary["similarity_vs_transition"] = similarity_artifact["summary"]
    summary["sequence_users"] = len(sequence_artifact["users"])
    write_json(diagnostics_dir / "diagnostics_summary.json", summary)
    logger.info("full diagnostics completed")
    return run_dir


def _resolve_diagnostics_config(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    dataset_section = config.get("dataset", {})
    if isinstance(dataset_section, dict) and dataset_section.get("config_path"):
        dataset_config = load_yaml_config(dataset_section["config_path"])
        merged = dict(dataset_config.get("dataset", {}))
        merged.update({key: value for key, value in dataset_section.items() if key != "config_path"})
        config["dataset"] = merged
    return config


def _prepare_run(config: dict[str, Any]) -> tuple[Path, Path, RunLogger]:
    experiment = config.get("experiment", {})
    seed = int(experiment.get("seed", config.get("dataset", {}).get("seed", 2026)))
    set_global_seed(seed)
    run_id = str(experiment.get("run_id", "phase2a_diagnostics"))
    output_root = ensure_dir(resolve_path(experiment.get("output_dir", "outputs/runs")))
    run_dir = ensure_dir(output_root / run_id)
    diagnostics_dir = ensure_dir(run_dir / "diagnostics")
    logger = RunLogger(run_dir / "logs.txt")
    save_resolved_config(config, run_dir / "resolved_config.yaml")
    write_json(run_dir / "environment.json", collect_environment())
    logger.info(f"starting diagnostics run_id={run_id}")
    return run_dir, diagnostics_dir, logger


def _prepare_processed_dataset(
    config: dict[str, Any],
    run_dir: Path,
    logger: RunLogger,
) -> Path:
    dataset = dict(config["dataset"])
    if bool(dataset.get("preprocess", True)):
        dataset["output_dir"] = str(run_dir / "artifacts" / "processed_dataset")
        logger.info("running preprocessing for diagnostics")
        return preprocess_from_config({"dataset": dataset}).output_dir
    return resolve_path(dataset["output_dir"])


def _diagnostics_summary(
    transition_edges: list[dict[str, Any]],
    time_window_edges: list[dict[str, Any]],
    item_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    item_catalog = {str(row["item_id"]) for row in item_rows}
    return {
        "item_count": len(item_catalog),
        "mean_transition_count": mean_transition_count(transition_edges),
        "time_window_edge_count": len(time_window_edges),
        "transition_coverage": transition_coverage(
            transition_edges,
            item_catalog=item_catalog,
        ),
        "transition_edge_count": len(transition_edges),
    }
