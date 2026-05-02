import json
import subprocess
import sys
from pathlib import Path

from llm4rec.io.artifacts import read_jsonl


def test_phase8_job_queue_is_planned_only():
    root = Path(__file__).resolve().parents[2]
    if not (root / "outputs/launch/paper_v1/jobs.jsonl").is_file():
        subprocess.run([sys.executable, "scripts/create_launch_manifest.py", "--output", "outputs/launch/paper_v1/launch_manifest.json"], cwd=root, check=True)
        subprocess.run([sys.executable, "scripts/create_job_queue.py", "--manifest", "outputs/launch/paper_v1/launch_manifest.json", "--output-dir", "outputs/launch/paper_v1"], cwd=root, check=True)
    jobs = read_jsonl(root / "outputs/launch/paper_v1/jobs.jsonl")
    assert jobs
    assert all(job["status"] == "planned" for job in jobs)
    assert all(job["allow_api_calls"] is False for job in jobs)
    assert all(job["NO_EXPERIMENTS_EXECUTED_IN_PHASE_8"] is True for job in jobs)
    manifest = json.loads((root / "outputs/launch/paper_v1/launch_manifest.json").read_text(encoding="utf-8"))
    assert manifest["NO_EXPERIMENTS_EXECUTED_IN_PHASE_8"] is True
