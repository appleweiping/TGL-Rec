import json
import subprocess
import sys
from pathlib import Path


def test_train_and_evaluate_smoke_commands():
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, "scripts/train.py", "--config", "configs/experiments/smoke.yaml"], cwd=root, check=True)
    subprocess.run([sys.executable, "scripts/evaluate.py", "--config", "configs/experiments/smoke.yaml"], cwd=root, check=True)
    run_dir = root / "outputs" / "runs" / "phase1_smoke"
    assert (run_dir / "checkpoints" / "mf_checkpoint.json").is_file()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["num_predictions"] > 0
