import subprocess
import sys
from pathlib import Path


def test_validate_experiment_script():
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [sys.executable, "scripts/validate_experiment.py", "--config", "configs/experiments/main_accuracy.yaml"],
        cwd=root,
        check=True,
    )
