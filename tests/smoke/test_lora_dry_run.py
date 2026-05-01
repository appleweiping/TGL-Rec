import json
import subprocess
import sys
from pathlib import Path


def test_lora_dry_run_writes_training_plan():
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, "scripts/train.py", "--config", "configs/training/lora.yaml", "--dry-run"], cwd=root, check=True)
    plan_path = root / "outputs" / "runs" / "lora_dry_run" / "training_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["dry_run"] is True
    assert plan["large_model_download_performed"] is False
