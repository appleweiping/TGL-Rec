import subprocess
import sys
from pathlib import Path


def test_phase7_protocol_validation_scripts():
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [sys.executable, "scripts/validate_experiment.py", "--config", "configs/experiments/phase7_pilot_movielens_sample.yaml"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [sys.executable, "scripts/validate_experiment.py", "--config", "configs/experiments/phase7_pilot_ablation_sample.yaml"],
        cwd=root,
        check=True,
    )
    subprocess.run([sys.executable, "scripts/validate_project.py"], cwd=root, check=True)
