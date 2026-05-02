import json
import subprocess
import sys
from pathlib import Path


def test_phase8_launch_preparation_scripts():
    root = Path(__file__).resolve().parents[2]
    commands = [
        [sys.executable, "scripts/check_dataset_readiness.py", "--config", "configs/datasets/movielens_full.yaml"],
        [sys.executable, "scripts/check_dataset_readiness.py", "--config", "configs/datasets/amazon_multidomain_full.yaml"],
        [sys.executable, "scripts/freeze_protocol.py", "--version", "protocol_v1", "--dry-run"],
        [sys.executable, "scripts/create_launch_manifest.py", "--output", "outputs/launch/paper_v1/launch_manifest.json"],
        [sys.executable, "scripts/create_job_queue.py", "--manifest", "outputs/launch/paper_v1/launch_manifest.json", "--output-dir", "outputs/launch/paper_v1"],
        [sys.executable, "scripts/estimate_paper_resources.py", "--manifest", "outputs/launch/paper_v1/launch_manifest.json"],
        [sys.executable, "scripts/plan_paper_tables.py", "--manifest", "outputs/launch/paper_v1/launch_manifest.json", "--output", "outputs/launch/paper_v1/table_plan.json"],
        [sys.executable, "scripts/check_launch_readiness.py", "--manifest", "outputs/launch/paper_v1/launch_manifest.json"],
        [sys.executable, "scripts/validate_experiment.py", "--config", "configs/experiments/paper_movielens_accuracy.yaml"],
        [sys.executable, "scripts/validate_experiment.py", "--config", "configs/experiments/paper_movielens_ablation.yaml"],
        [sys.executable, "scripts/validate_experiment.py", "--config", "configs/experiments/paper_amazon_multidomain_accuracy.yaml"],
        [sys.executable, "scripts/validate_project.py"],
    ]
    for command in commands:
        subprocess.run(command, cwd=root, check=True)
    report = json.loads((root / "outputs/launch/paper_v1/validation/launch_readiness.json").read_text(encoding="utf-8"))
    assert report["NO_EXPERIMENTS_EXECUTED_IN_PHASE_8"] is True
    assert report["status"] in {"GO", "CONDITIONAL_GO"}
