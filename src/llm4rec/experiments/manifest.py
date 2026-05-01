"""Experiment manifest loading and validation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REQUIRED_EXPERIMENT_FIELDS = [
    "dataset",
    "split_strategy",
    "candidate_strategy",
    "methods",
    "seeds",
    "metrics",
    "output_dir",
    "run_mode",
    "reportable",
]


@dataclass(frozen=True)
class ExperimentManifest:
    """Resolved high-level experiment manifest."""

    data: dict[str, Any]

    @property
    def reportable(self) -> bool:
        return bool(self.data.get("reportable", False))


def manifest_from_config(config: dict[str, Any]) -> ExperimentManifest:
    """Extract a manifest section or compatible top-level config."""

    if "manifest" in config:
        return ExperimentManifest(dict(config["manifest"]))
    experiment = dict(config.get("experiment", {}))
    return ExperimentManifest(
        {
            "candidate_strategy": config.get("candidate_strategy", config.get("dataset", {}).get("candidate_protocol")),
            "dataset": config.get("dataset", {}).get("name", config.get("dataset")),
            "methods": config.get("methods", []),
            "metrics": config.get("evaluation", {}).get("metrics", config.get("metrics", [])),
            "output_dir": experiment.get("output_dir"),
            "reportable": experiment.get("reportable", config.get("reportable", False)),
            "run_mode": experiment.get("run_mode"),
            "seeds": experiment.get("seeds", [experiment.get("seed")]),
            "split_strategy": config.get("dataset", {}).get("split_strategy", config.get("split_strategy")),
        }
    )
