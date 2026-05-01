import subprocess
import sys
from pathlib import Path


def test_export_tables_from_metrics():
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, "scripts/run_experiment.py", "--config", "configs/experiments/smoke.yaml"], cwd=root, check=True)
    subprocess.run([sys.executable, "scripts/export_tables.py", "--input", "outputs/runs", "--output", "outputs/tables"], cwd=root, check=True)
    assert (root / "outputs" / "tables" / "paper_table.csv").is_file()
    assert (root / "outputs" / "tables" / "table_manifest.json").is_file()
