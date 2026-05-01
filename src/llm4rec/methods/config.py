"""Method config loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.experiments.config import deep_merge, load_yaml_config


DEFAULT_METHOD_CONFIG: dict[str, Any] = {
    "method": {
        "name": "time_graph_evidence_rec",
        "codename": "TimeGraphEvidenceRec",
        "reportable": False,
        "eval_split": "test",
    },
    "retrieval": {
        "modes": [
            "transition_topk",
            "time_window_topk",
            "semantic_topk",
            "contrastive_transition_only",
            "recent_history_focused",
            "user_drift_blocks",
        ],
        "top_k_per_candidate": 4,
        "recent_history_items": 2,
        "semantic_min_similarity": 0.1,
        "contrastive_max_similarity": 0.35,
    },
    "translator": {"mode": "prompt_ready_json"},
    "scoring": {
        "transition_weight": 1.0,
        "window_weight": 0.5,
        "semantic_weight": 0.25,
        "recency_weight": 0.1,
    },
    "time_window": {"window_seconds": 86400},
    "encoder": {"type": "temporal_memory_stub", "reportable": False},
    "ablation": {},
}


def load_method_config(path_or_config: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load and merge a method config with Phase 5 defaults."""

    if isinstance(path_or_config, (str, Path)):
        raw = load_yaml_config(path_or_config)
    else:
        raw = dict(path_or_config)
    return deep_merge(DEFAULT_METHOD_CONFIG, raw)


def resolve_method_from_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve the method section from an experiment config."""

    section = dict(config.get("method", {}))
    if section.get("config_path"):
        base = load_method_config(section["config_path"])
        override = {key: value for key, value in section.items() if key != "config_path"}
        return deep_merge(base, {"method": override})
    return load_method_config(config.get("method_config", section))
