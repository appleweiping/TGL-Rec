"""Phase 9C multi-seed protocol-v1 paper matrix orchestration."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from llm4rec.evaluation.aggregate import aggregate_multiseed_results
from llm4rec.experiments.config import resolve_path
from llm4rec.experiments.paper_matrix import PaperMatrixRequest, normalize_method, run_paper_matrix
from llm4rec.io.artifacts import ensure_dir, write_json


def run_multiseed_paper_matrix(
    *,
    manifest_path: str | Path,
    matrix: str,
    seeds: list[int],
    datasets: list[str],
    methods: list[str],
    output_dir: str | Path,
    candidate_output_mode: str,
    shared_pool_scoring: bool,
    continue_on_failure: bool,
    max_expanded_candidate_items: int = 200,
    scoring_config_path: str | Path = "configs/scoring/shared_pool.yaml",
) -> Path:
    """Run new nonzero seeds for the Phase 9C main accuracy matrix."""

    root = ensure_dir(resolve_path(output_dir))
    normalized_methods = [normalize_method(str(method)) for method in methods]
    request_manifest = {
        "api_calls_allowed": False,
        "candidate_output_mode": candidate_output_mode,
        "continue_on_failure": bool(continue_on_failure),
        "datasets": [str(dataset) for dataset in datasets],
        "llm_provider": "none",
        "lora_training_enabled": False,
        "matrix": str(matrix),
        "methods": normalized_methods,
        "protocol_version": "protocol_v1",
        "seed0_policy": "reuse_existing_outputs_only",
        "seeds_requested": [int(seed) for seed in seeds],
        "shared_pool_scoring": bool(shared_pool_scoring),
    }
    write_json(root / "run_manifest.json", request_manifest)
    statuses: list[dict[str, Any]] = []
    for seed in seeds:
        if int(seed) == 0:
            statuses.append({"message": "seed 0 must be aggregated from existing outputs", "seed": 0, "status": "skipped"})
            continue
        started = time.perf_counter()
        seed_dir = root / f"seed_{int(seed)}"
        try:
            run_paper_matrix(
                PaperMatrixRequest(
                    manifest_path=Path(manifest_path),
                    matrix=str(matrix),
                    seed=int(seed),
                    datasets=tuple(str(dataset) for dataset in datasets),
                    methods=tuple(normalized_methods),
                    output_dir=seed_dir,
                    continue_on_failure=bool(continue_on_failure),
                    candidate_output_mode="compact_ref_v1"
                    if str(candidate_output_mode) == "compact_ref"
                    else str(candidate_output_mode),
                    max_expanded_candidate_items=int(max_expanded_candidate_items),
                    rerun_failed_only=False,
                    shared_pool_scoring=bool(shared_pool_scoring),
                    scoring_config_path=Path(scoring_config_path),
                )
            )
            statuses.append(
                {
                    "runtime_seconds": time.perf_counter() - started,
                    "seed": int(seed),
                    "seed_dir": str(seed_dir),
                    "status": "succeeded",
                }
            )
        except Exception as exc:
            statuses.append(
                {
                    "failure_reason": type(exc).__name__,
                    "message": str(exc),
                    "runtime_seconds": time.perf_counter() - started,
                    "seed": int(seed),
                    "seed_dir": str(seed_dir),
                    "status": "failed",
                }
            )
            write_json(seed_dir / "failure_report.json", statuses[-1])
            if not continue_on_failure:
                write_json(root / "seed_run_status.json", {"seeds": statuses})
                raise
    write_json(root / "seed_run_status.json", {"seeds": statuses})
    return root


def aggregate_phase9c_outputs(
    *,
    seed0_dir: str | Path,
    multiseed_dir: str | Path,
    seeds: list[int],
) -> dict[str, Any]:
    """Aggregate Phase 9C outputs across seed 0 and newly run seeds."""

    return aggregate_multiseed_results(
        seed0_dir=seed0_dir,
        multiseed_dir=multiseed_dir,
        seeds=[int(seed) for seed in seeds],
    )
