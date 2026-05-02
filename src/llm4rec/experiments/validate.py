"""Experiment and project validation for pre-experiment readiness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.data.readiness import NO_EXECUTION_FLAG
from llm4rec.experiments.config import load_yaml_config, resolve_path
from llm4rec.experiments.manifest import REQUIRED_EXPERIMENT_FIELDS, manifest_from_config
from llm4rec.experiments.protocol_version import DEFAULT_PAPER_CONFIGS


class ExperimentValidationError(ValueError):
    """Raised when an experiment config is not ready."""


PAPER_ALLOWED_METHODS = {
    "bm25",
    "full",
    "graph_only",
    "mf",
    "popularity",
    "sasrec",
    "temporal_graph_encoder",
    "text_only",
    "time_graph_evidence",
    "time_graph_evidence_dynamic",
    "w_o_dynamic_encoder",
    "w_o_semantic_similarity",
    "w_o_temporal_graph",
    "w_o_time_gap_tags",
    "w_o_time_window_edges",
    "w_o_transition_edges",
}


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
    experiment = dict(config.get("experiment", {}))
    training = dict(config.get("training", {}))
    dataset = dict(config.get("dataset", {}))
    evaluation = dict(config.get("evaluation", {}))
    pilot = dict(config.get("pilot", {}))
    if data.get("run_mode") == "pilot" or experiment.get("run_mode") == "pilot":
        if bool(config.get("pilot_reportable", False)):
            errors.append("pilot configs must set pilot_reportable=false")
        if bool(llm.get("allow_api_calls", False)):
            errors.append("pilot configs cannot allow API calls")
        if bool(training.get("enable_lora_training", False)):
            errors.append("pilot configs cannot enable LoRA training")
        if data.get("candidate_strategy") != evaluation.get("candidate_protocol", dataset.get("candidate_protocol")):
            errors.append("pilot methods must share the manifest/evaluation candidate protocol")
        if pilot.get("split_artifact") != "shared":
            errors.append("pilot configs must declare shared split_artifact")
        if pilot.get("candidate_artifact") != "shared":
            errors.append("pilot configs must declare shared candidate_artifact")
    if manifest.reportable and data.get("run_mode") == "pilot" and not bool(config.get("allow_reportable_pilot_data", False)):
        errors.append("reportable configs cannot use sampled pilot data unless explicitly allowed")
    if manifest.reportable and llm.get("provider") == "mock":
        errors.append("reportable configs cannot use mock LLM")
    if manifest.reportable and bool(llm.get("allow_api_calls", False)):
        errors.append("reportable configs cannot enable API calls by default")
    if data.get("run_mode") == "paper" or experiment.get("run_mode") == "paper":
        _validate_paper_config(config, data, methods, errors)
    if errors:
        raise ExperimentValidationError("; ".join(errors))
    return {"config_path": str(resolve_path(config_path)), "manifest": data, "status": "pass"}


def validate_project(root: str | Path = ".") -> dict[str, Any]:
    """Validate required project files, safety defaults, and launch package when present."""

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
        "docs/frozen_experiment_protocol.md",
        "docs/pilot_protocol.md",
        "docs/reportable_rules.md",
        "docs/failure_modes.md",
        "docs/resource_budget.md",
        "docs/paper_experiment_matrix.md",
        "configs/experiments/phase7_pilot_movielens_sample.yaml",
        "configs/experiments/phase7_pilot_ablation_sample.yaml",
        "configs/experiments/phase7_pilot_resource_check.yaml",
        "scripts/estimate_pilot_resources.py",
        "scripts/run_pilot_matrix.py",
        "scripts/export_pilot_tables.py",
        "scripts/audit_failures.py",
        "src/llm4rec/experiments/pilot_runner.py",
        "src/llm4rec/experiments/resource_estimator.py",
        "src/llm4rec/evaluation/pilot_export.py",
        "src/llm4rec/evaluation/failure_audit.py",
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
        "configs/datasets/movielens_full.yaml",
        "configs/datasets/amazon_reviews_2023.yaml",
        "configs/datasets/amazon_multidomain_full.yaml",
        "configs/datasets/amazon_multidomain_sampled.yaml",
        "configs/datasets/amazon_multidomain_filtered_k3.yaml",
        "configs/datasets/amazon_multidomain_filtered_k5.yaml",
        "configs/datasets/amazon_multidomain_filtered_iterative_k3.yaml",
        "configs/datasets/amazon_multidomain_filtered_iterative_k5.yaml",
        "configs/experiments/paper_movielens_accuracy.yaml",
        "configs/experiments/paper_movielens_ablation.yaml",
        "configs/experiments/paper_movielens_long_tail.yaml",
        "configs/experiments/paper_movielens_efficiency.yaml",
        "configs/experiments/paper_amazon_multidomain_accuracy.yaml",
        "configs/experiments/paper_amazon_multidomain_ablation.yaml",
        "configs/experiments/paper_amazon_multidomain_cold_start.yaml",
        "configs/experiments/paper_amazon_multidomain_efficiency.yaml",
        "docs/dataset_readiness.md",
        "docs/amazon_data_setup.md",
        "docs/amazon_filtering_policy.md",
        "docs/protocol_versions.md",
        "docs/paper_table_plan.md",
        "scripts/check_dataset_readiness.py",
        "scripts/check_amazon_schema.py",
        "scripts/freeze_data_artifacts.py",
        "scripts/prepare_amazon_multidomain.py",
        "scripts/filter_amazon_multidomain.py",
        "scripts/freeze_protocol.py",
        "scripts/create_launch_manifest.py",
        "scripts/create_job_queue.py",
        "scripts/estimate_paper_resources.py",
        "scripts/plan_paper_tables.py",
        "scripts/lock_results.py",
        "scripts/check_launch_readiness.py",
        "src/llm4rec/data/readiness.py",
        "src/llm4rec/data/artifact_freeze.py",
        "src/llm4rec/data/amazon_reviews_2023.py",
        "src/llm4rec/data/amazon_converter.py",
        "src/llm4rec/data/amazon_filtering.py",
        "src/llm4rec/data/filtering.py",
        "src/llm4rec/data/kcore.py",
        "src/llm4rec/data/schema_validation.py",
        "src/llm4rec/experiments/protocol_version.py",
        "src/llm4rec/experiments/artifact_registry.py",
        "src/llm4rec/experiments/launch_manifest.py",
        "src/llm4rec/experiments/job_queue.py",
        "src/llm4rec/experiments/resume.py",
        "src/llm4rec/experiments/resource_budget.py",
        "src/llm4rec/evaluation/result_lock.py",
        "src/llm4rec/evaluation/table_plan.py",
    ]
    missing = [path for path in required if not (repo / path).exists()]
    llm_config = load_yaml_config(repo / "configs/llm/openai_compatible.yaml")
    errors = list(missing)
    if bool(llm_config.get("llm", {}).get("allow_api_calls", False)):
        errors.append("configs/llm/openai_compatible.yaml must default allow_api_calls=false")
    gitignore = (repo / ".gitignore").read_text(encoding="utf-8") if (repo / ".gitignore").is_file() else ""
    if "outputs/" not in gitignore:
        errors.append("outputs/ must be gitignored")
    pilot_output_checks = _pilot_output_checks(repo)
    errors.extend(pilot_output_checks["errors"])
    if (repo / "outputs/launch/paper_v1").exists():
        for config_path in DEFAULT_PAPER_CONFIGS:
            try:
                validate_experiment_config(repo / config_path)
            except Exception as exc:
                errors.append(f"{config_path}: {exc}")
    launch_checks = _launch_output_checks(repo)
    errors.extend(launch_checks["errors"])
    if errors:
        raise ExperimentValidationError("; ".join(errors))
    optional_dependencies = {"torch": _dependency_available("torch")}
    return {
        "checked_files": required,
        "launch_output_checks": launch_checks,
        "optional_dependencies": optional_dependencies,
        "pilot_output_checks": pilot_output_checks,
        "status": "pass",
    }


def _validate_paper_config(
    config: dict[str, Any],
    manifest_data: dict[str, Any],
    methods: list[Any],
    errors: list[str],
) -> None:
    if not bool(manifest_data.get("reportable", False)):
        errors.append("paper configs must set reportable=true")
    if config.get("protocol_version") in (None, ""):
        errors.append("paper configs must declare protocol_version")
    if bool(config.get("api_calls_allowed", False)):
        errors.append("paper configs cannot allow API calls")
    if bool(config.get("lora_training_enabled", False)):
        errors.append("paper configs cannot enable LoRA training")
    if bool(config.get("training", {}).get("enable_lora_training", False)):
        errors.append("paper configs cannot enable LoRA training")
    if bool(config.get("llm", {}).get("allow_api_calls", False)):
        errors.append("paper configs cannot allow API calls")
    if config.get("split_artifact") in (None, ""):
        errors.append("paper configs must declare split_artifact")
    if config.get("candidate_artifact") in (None, ""):
        errors.append("paper configs must declare candidate_artifact")
    forbidden_paths = [
        str(manifest_data.get("output_dir", "")),
        str(config.get("split_artifact", "")),
        str(config.get("candidate_artifact", "")),
    ]
    if any("phase7" in value or "pilot" in value for value in forbidden_paths):
        errors.append("paper configs cannot use pilot output paths")
    for method in methods:
        text = str(method.get("name", method)) if isinstance(method, dict) else str(method)
        lowered = text.lower()
        if lowered not in PAPER_ALLOWED_METHODS:
            errors.append(f"unknown or unapproved paper method: {text}")
        if "mock" in lowered or "stub" in lowered or "skeleton" in lowered or "markov" in lowered:
            errors.append("paper configs cannot use mock/stub/skeleton/Markov methods")
    readiness_path = _readiness_path_for_dataset(str(manifest_data.get("dataset", "")))
    if not readiness_path.is_file():
        errors.append(f"dataset readiness status missing: {readiness_path}")


def _dependency_available(module_name: str) -> bool:
    try:
        __import__(module_name)
    except Exception:
        return False
    return True


def _pilot_output_checks(repo: Path) -> dict[str, Any]:
    checks = {
        "ablation_failure_report": False,
        "ablation_non_reportable_table": False,
        "main_failure_report": False,
        "main_non_reportable_table": False,
        "resource_estimate": False,
    }
    errors: list[str] = []
    main = repo / "outputs/runs/phase7_pilot_movielens_sample"
    ablation = repo / "outputs/runs/phase7_pilot_ablation_sample"
    if main.exists():
        checks["resource_estimate"] = (main / "resource_estimate.json").is_file()
        checks["main_failure_report"] = (main / "failure_report.json").is_file()
        table = main / "pilot_table.csv"
        checks["main_non_reportable_table"] = table.is_file() and "NON_REPORTABLE" in table.read_text(encoding="utf-8")
        if not checks["resource_estimate"]:
            errors.append("phase7 pilot resource_estimate.json missing")
        if not checks["main_failure_report"]:
            errors.append("phase7 pilot failure_report.json missing")
        if not checks["main_non_reportable_table"]:
            errors.append("phase7 pilot table missing NON_REPORTABLE marker")
    if ablation.exists():
        checks["ablation_failure_report"] = (ablation / "failure_report.json").is_file()
        table = ablation / "ablation_table.csv"
        checks["ablation_non_reportable_table"] = table.is_file() and "NON_REPORTABLE" in table.read_text(encoding="utf-8")
        if not checks["ablation_failure_report"]:
            errors.append("phase7 ablation failure_report.json missing")
        if not checks["ablation_non_reportable_table"]:
            errors.append("phase7 ablation table missing NON_REPORTABLE marker")
    return {"checks": checks, "errors": errors}


def _launch_output_checks(repo: Path) -> dict[str, Any]:
    launch = repo / "outputs/launch/paper_v1"
    checks = {
        "go_no_go_checklist": False,
        "job_queue": False,
        "launch_manifest": False,
        "launch_readiness": False,
        "no_jobs_executed": False,
        "protocol_manifest": False,
        "resource_budget": False,
        "table_plan": False,
    }
    errors: list[str] = []
    if not launch.exists():
        return {"checks": checks, "errors": errors}
    required = {
        "go_no_go_checklist": launch / "go_no_go_checklist.md",
        "job_queue": launch / "jobs.jsonl",
        "launch_manifest": launch / "launch_manifest.json",
        "launch_readiness": launch / "validation" / "launch_readiness.json",
        "protocol_manifest": launch / "protocol" / "protocol_manifest.json",
        "resource_budget": launch / "resource_budget.json",
        "table_plan": launch / "table_plan.json",
    }
    for key, path in required.items():
        checks[key] = path.is_file()
        if not checks[key]:
            errors.append(f"missing Phase 8 launch artifact: {path}")
    jobs_path = launch / "jobs.jsonl"
    if jobs_path.is_file():
        import json

        statuses = []
        for line in jobs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            statuses.append(row.get("status"))
            if row.get("allow_api_calls") is not False:
                errors.append("Phase 8 job queue contains an API-enabled job")
                break
            if row.get(NO_EXECUTION_FLAG) is not True:
                errors.append("Phase 8 job queue missing no-execution flag")
                break
        checks["no_jobs_executed"] = all(status == "planned" for status in statuses) if statuses else False
        if not checks["no_jobs_executed"]:
            errors.append("Phase 8 job queue contains executed/non-planned jobs")
    return {"checks": checks, "errors": errors}


def _readiness_path_for_dataset(dataset_name: str) -> Path:
    if dataset_name == "movielens_full":
        return resolve_path("outputs/launch/paper_v1/dataset_readiness/movielens_full_readiness.json")
    if dataset_name == "amazon_multidomain_full":
        return resolve_path("outputs/launch/paper_v1/dataset_readiness/amazon_multidomain_full_readiness.json")
    if dataset_name == "amazon_multidomain_filtered_iterative_k3":
        return resolve_path(
            "outputs/launch/paper_v1/dataset_readiness/amazon_multidomain_filtered_iterative_k3_readiness.json"
        )
    return resolve_path(f"outputs/launch/paper_v1/dataset_readiness/{dataset_name}_readiness.json")
