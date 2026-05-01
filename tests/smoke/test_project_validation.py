import subprocess
import sys
from pathlib import Path


def test_project_validation_script():
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, "scripts/validate_project.py"], cwd=root, check=True)
