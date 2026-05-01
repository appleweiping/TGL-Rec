import json

from llm4rec.experiments.pilot_runner import run_pilot_matrix


def test_pilot_manifest_records_shared_protocol():
    run_dir = run_pilot_matrix("configs/experiments/phase7_pilot_movielens_sample.yaml")
    manifest = json.loads((run_dir / "pilot_manifest.json").read_text(encoding="utf-8"))
    assert manifest["NON_REPORTABLE"] == "NON_REPORTABLE"
    assert manifest["candidate_protocol"] == "sampled_fixed"
    assert manifest["split_artifact"].endswith("interactions.jsonl")
