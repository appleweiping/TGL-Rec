"""Server run planning for the large four-domain same-candidate protocol."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from llm4rec.io.artifacts import ensure_dir, write_json


DEFAULT_DOMAINS = ("beauty", "books", "electronics", "movies")
DEFAULT_SPLITS = ("valid", "test")
DEFAULT_PROTOCOL_VERSION = "protocol_week8_large10000_same_candidate"


@dataclass(frozen=True)
class PlannedTaskDir:
    """One expected external same-candidate task directory."""

    domain: str
    split: str
    path: str
    exists: bool


def build_four_domain_server_plan(
    *,
    external_root: str | Path,
    protocol_version: str = DEFAULT_PROTOCOL_VERSION,
    domains: Iterable[str] = DEFAULT_DOMAINS,
    splits: Iterable[str] = DEFAULT_SPLITS,
    base_model_path: str = "/home/ajifang/models/Qwen/Qwen3-8B",
    include_missing_task_dirs: bool = False,
) -> dict[str, object]:
    """Build a non-executing command plan for server-side large-domain work."""

    root = Path(external_root).expanduser()
    tasks = [
        _planned_task(root, domain=str(domain), split=str(split))
        for domain in domains
        for split in splits
    ]
    importable = [task for task in tasks if task.exists or include_missing_task_dirs]
    missing = [task for task in tasks if not task.exists]
    return {
        "base_model_path": base_model_path,
        "domains": [str(domain) for domain in domains],
        "external_root": str(root),
        "milestone": "phase10_four_domain_same_candidate",
        "missing_task_dirs": [asdict(task) for task in missing],
        "planned_task_dirs": [asdict(task) for task in tasks],
        "protocol_version": protocol_version,
        "run_order": [
            "server_sync",
            "external_task_inventory",
            "import_frozen_same_candidate_tasks",
            "artifact_readiness_checks",
            "build_week8_lora_sft",
            "merge_week8_lora_sft",
            "train_week8_lora_controls",
            "evaluate_week8_lora_controls",
            "control_lora_diagnostic_eval",
            "official_baseline_adaptation",
            "ours_framework_ablation_matrix",
            "reviewer_gate_before_paper_claims",
        ],
        "commands": {
            "server_sync": [
                "cd ~/projects/TGL-Rec",
                "git pull",
                "conda activate qwen_vllm",
            ],
            "external_task_inventory": [
                f"find {root} -path \"*large10000_100neg*\" -type f | sort",
            ],
            "import_frozen_same_candidate_tasks": _import_commands(
                importable,
                protocol_version=protocol_version,
            ),
            "artifact_readiness_checks": _artifact_check_commands(
                domains=[str(domain) for domain in domains],
                protocol_version=protocol_version,
            ),
            "build_week8_lora_sft": [
                "python scripts/build_lora_sft_data.py "
                "--config configs/experiments/week8_lora_8b_history_only.yaml --materialize",
                "python scripts/build_lora_sft_data.py "
                "--config configs/experiments/week8_lora_8b_temporal_evidence.yaml --materialize",
            ],
            "merge_week8_lora_sft": _sft_merge_commands(
                domains=[str(domain) for domain in domains],
                protocol_version=protocol_version,
            ),
            "train_week8_lora_controls": [
                "mkdir -p outputs/paper_runs/protocol_week8_large10000_same_candidate/lora_8b/history_only_sft "
                "outputs/paper_runs/protocol_week8_large10000_same_candidate/lora_8b/temporal_evidence_sft",
                "CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_lora_8b.py "
                "--config configs/experiments/week8_lora_8b_history_only.yaml "
                "> outputs/paper_runs/protocol_week8_large10000_same_candidate/lora_8b/history_only_sft/train.nohup.log 2>&1 &",
                "CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_lora_8b.py "
                "--config configs/experiments/week8_lora_8b_temporal_evidence.yaml "
                "> outputs/paper_runs/protocol_week8_large10000_same_candidate/lora_8b/temporal_evidence_sft/train.nohup.log 2>&1 &",
            ],
            "evaluate_week8_lora_controls": [
                "CUDA_VISIBLE_DEVICES=0 python -u scripts/run_lora_rerank_eval.py "
                "--config configs/experiments/week8_lora_8b_rerank_eval.yaml "
                f"--base-model-path {base_model_path}",
                "test -f outputs/paper_runs/protocol_week8_large10000_same_candidate/lora_8b/eval/predictions.jsonl && "
                "python scripts/diagnose_lora_predictions.py "
                "--predictions outputs/paper_runs/protocol_week8_large10000_same_candidate/lora_8b/eval/predictions.jsonl "
                "--output-dir outputs/paper_runs/protocol_week8_large10000_same_candidate/lora_8b/eval/diagnostics",
            ],
            "control_lora_diagnostic_eval": [
                "CUDA_VISIBLE_DEVICES=0 python -u scripts/run_lora_rerank_eval.py "
                "--config configs/experiments/paper_lora_8b_rerank_eval.yaml "
                f"--base-model-path {base_model_path} --limit 20 --top-m 50",
                "test -f outputs/paper_runs/protocol_v1/lora_8b/eval/predictions.jsonl && "
                "python scripts/diagnose_lora_predictions.py "
                "--predictions outputs/paper_runs/protocol_v1/lora_8b/eval/predictions.jsonl "
                "--output-dir outputs/paper_runs/protocol_v1/lora_8b/eval/diagnostics",
            ],
        },
        "safety_rules": [
            "Do not resample users, positives, negatives, or candidates.",
            "Do not overwrite protocol_v1; import large tasks under a new protocol version.",
            "Do not run diagnostics unless predictions.jsonl exists.",
            "Do not merge scaffold or non-reportable baselines into main paper tables.",
        ],
    }


def _planned_task(root: Path, *, domain: str, split: str) -> PlannedTaskDir:
    path = root / f"{domain}_large10000_100neg_{split}_same_candidate"
    return PlannedTaskDir(domain=domain, split=split, path=str(path), exists=path.is_dir())


def _import_commands(
    tasks: list[PlannedTaskDir],
    *,
    protocol_version: str,
) -> list[str]:
    if not tasks:
        return [
            "echo 'No existing task dirs were found. Re-run with --include-missing-task-dirs "
            "only after confirming the external project has produced the expected directories.'"
        ]
    task_args = " ".join(f"--task-dir {task.path}" for task in tasks)
    return [
        "python scripts/import_week8_same_candidate.py "
        f"{task_args} --protocol-version {protocol_version}"
    ]


def _artifact_check_commands(
    *,
    domains: list[str],
    protocol_version: str,
) -> list[str]:
    commands = []
    for domain in domains:
        root = f"outputs/artifacts/{protocol_version}/{domain}"
        commands.extend(
            [
                f"test -f {root}/candidates.jsonl",
                f"test -f {root}/splits.jsonl",
                f"test -f {root}/item_metadata.jsonl",
                f"test -f {root}/latest_import_manifest.json",
            ]
        )
    return commands


def _sft_merge_commands(
    *,
    domains: list[str],
    protocol_version: str,
) -> list[str]:
    input_root = f"data/processed/lora_sft/{protocol_version}"
    return [
        "python scripts/merge_lora_sft_data.py "
        f"--input-root {input_root} "
        f"--output-dir {input_root}/four_domain/history_only_sft "
        "--variant history_only_sft "
        f"--datasets {' '.join(domains)} "
        f"--protocol-version {protocol_version}",
        "python scripts/merge_lora_sft_data.py "
        f"--input-root {input_root} "
        f"--output-dir {input_root}/four_domain/temporal_evidence_sft "
        "--variant temporal_evidence_sft "
        f"--datasets {' '.join(domains)} "
        f"--protocol-version {protocol_version}",
    ]


def write_shell_runbook(plan: dict[str, object], output_path: str | Path) -> Path:
    """Write a shell runbook from a generated plan."""

    path = Path(output_path)
    ensure_dir(path.parent)
    commands = plan.get("commands", {})
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by scripts/plan_four_domain_runs.py. Inspect before running.",
    ]
    if isinstance(commands, dict):
        for section, section_commands in commands.items():
            lines.extend(["", f"echo '== {section} =='"])
            if isinstance(section_commands, list):
                lines.extend(str(command) for command in section_commands)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--protocol-version", default=DEFAULT_PROTOCOL_VERSION)
    parser.add_argument("--domain", action="append", dest="domains")
    parser.add_argument("--split", action="append", dest="splits")
    parser.add_argument("--base-model-path", default="/home/ajifang/models/Qwen/Qwen3-8B")
    parser.add_argument("--include-missing-task-dirs", action="store_true")
    parser.add_argument("--output", default="outputs/plans/four_domain_server_plan.json")
    parser.add_argument("--shell-output", default="")
    args = parser.parse_args(argv)

    plan = build_four_domain_server_plan(
        external_root=args.external_root,
        protocol_version=args.protocol_version,
        domains=args.domains or DEFAULT_DOMAINS,
        splits=args.splits or DEFAULT_SPLITS,
        base_model_path=args.base_model_path,
        include_missing_task_dirs=args.include_missing_task_dirs,
    )
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    write_json(output_path, plan)
    if args.shell_output:
        write_shell_runbook(plan, args.shell_output)
    print(json.dumps({"output": str(output_path), "status": "succeeded"}, sort_keys=True))
    return 0
