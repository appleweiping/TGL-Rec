"""Experiment and project validation for pre-experiment readiness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.experiments.config import load_yaml_config, resolve_path
from llm4rec.experiments.manifest import REQUIRED_EXPERIMENT_FIELDS, manifest_from_config


class ExperimentValidationError(ValueError):
    """Raised when an experiment config is not ready."""


def validate_experiment_config(config_path: str | Path) -> dict[str, Any]:
    """Validate manifest completeness and safety constraints."""

    config = load_yaml_config(config_path)
    manifest = manifest_from_config(config)
    data = manifest.data
    errors: list[str] = []
    for field in REQUIRED_EXPERIMENT_FIELDS:
        if data.get(field) in (None, "", []):
            errors.append(f"missing manifest field: {field}")
    methods = data.get("methods", [])
    if manifest.reportable:
        if data.get("run_mode") in {"smoke", "diagnostic_mock"}:
            errors.append("reportable configs cannot use smoke or diagnostic_mock run_mode")
        for method in methods:
            text = str(method).lower()
            if "mock" in text or "stub" in text or "skeleton" in text:
                errors.append("reportable configs cannot use mock/stub/skeleton methods")
    for method in methods:
        if isinstance(method, str) and (method.endswith(".yaml") or "/" in method):
            path = resolve_path(method)
            if not path.is_file():
                errors.append(f"missing baseline config: {method}")
    llm = dict(config.get("llm", {}))
    if manifest.reportable and llm.get("provider") == "mock":
        errors.append("reportable configs cannot use mock LLM")
    if manifest.reportable and bool(llm.get("allow_api_calls", False)):
        errors.append("reportable configs cannot enable API calls by default")
    if errors:
        raise ExperimentValidationError("; ".join(errors))
    return {"config_path": str(resolve_path(config_path)), "manifest": data, "status": "pass"}


def validate_project(root: str | Path = ".") -> dict[str, Any]:
    """Validate required Phase 4 project files and safety defaults."""

    repo = Path(root).resolve()
    required = [
        "configs/diagnostics/llm_sequence_time_api_micro.yaml",
        "configs/experiments/main_accuracy.yaml",
        "configs/experiments/ablation.yaml",
        "configs/llm/openai_compatible.yaml",
        "docs/RESEARCH_IDEA.md",
        "docs/baselines.md",
        "docs/data_format.md",
        "docs/evaluation_metrics.md",
        "docs/experiment_protocol.md",
        "docs/phase_plan.md",
        "docs/pre_experiment_checklist.md",
        "docs/reproducibility.md",
        "scripts/train.py",
        "scripts/evaluate.py",
        "scripts/export_tables.py",
        "scripts/export_method_card.py",
        "scripts/run_method_smoke.py",
        "scripts/train_sasrec.py",
        "scripts/train_temporal_graph.py",
        "scripts/validate_experiment.py",
        "configs/encoders/temporal_graph.yaml",
        "configs/methods/time_graph_evidence.yaml",
        "configs/methods/time_graph_evidence_ablation.yaml",
        "configs/experiments/phase5_method_smoke.yaml",
        "configs/experiments/phase5_ablation_smoke.yaml",
        "configs/experiments/phase6_sasrec_smoke.yaml",
        "configs/experiments/phase6_temporal_graph_smoke.yaml",
        "configs/experiments/phase6_method_encoder_smoke.yaml",
        "configs/training/sasrec_smoke.yaml",
        "configs/training/temporal_graph_smoke.yaml",
        "docs/method_card_time_graph_evidence.md",
        "docs/oursmethod_ablation_plan.md",
        "docs/leakage_protocol.md",
        "docs/sequential_baselines.md",
        "docs/dynamic_graph_encoder.md",
        "src/llm4rec/evidence/base.py",
        "src/llm4rec/evidence/retriever.py",
        "src/llm4rec/evidence/translator.py",
        "src/llm4rec/methods/time_graph_evidence.py",
        "src/llm4rec/methods/leakage.py",
        "src/llm4rec/models/sasrec.py",
        "src/llm4rec/trainers/sasrec.py",
        "src/llm4rec/trainers/temporal_graph.py",
        "src/llm4rec/rankers/time_graph_evidence_ranker.py",
        "src/llm4rec/rankers/sasrec.py",
        "src/llm4rec/rankers/temporal_graph.py",
        "src/llm4rec/encoders/base.py",
        "src/llm4rec/encoders/temporal_graph_encoder.py",
    ]
    missing = [path for path in required if not (repo / path).exists()]
    llm_config = load_yaml_config(repo / "configs/llm/openai_compatible.yaml")
    errors = list(missing)
    if bool(llm_config.get("llm", {}).get("allow_api_calls", False)):
        errors.append("configs/llm/openai_compatible.yaml must default allow_api_calls=false")
    gitignore = (repo / ".gitignore").read_text(encoding="utf-8") if (repo / ".gitignore").is_file() else ""
    if "outputs/" not in gitignore:
        errors.append("outputs/ must be gitignored")
    if errors:
        raise ExperimentValidationError("; ".join(errors))
    optional_dependencies = {"torch": _dependency_available("torch")}
    return {
        "checked_files": required,
        "optional_dependencies": optional_dependencies,
        "status": "pass",
    }


def _dependency_available(module_name: str) -> bool:
    try:
        __import__(module_name)
    except Exception:
        return False
    return True
