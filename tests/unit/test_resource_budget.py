from llm4rec.data.readiness import NO_EXECUTION_FLAG
from llm4rec.experiments.launch_manifest import create_launch_manifest
from llm4rec.experiments.resource_budget import estimate_paper_resources


def test_paper_resource_budget_forbids_api_and_lora(tmp_path):
    manifest = create_launch_manifest(tmp_path / "launch_manifest.json")
    budget = estimate_paper_resources(manifest, tmp_path / "resource_budget.json")
    assert budget[NO_EXECUTION_FLAG] is True
    assert budget["jobs"] == manifest["total_planned_runs"]
    assert budget["api_calls"] == 0
    assert budget["lora_training_jobs"] == 0
    assert budget["trainable_jobs"] > 0
