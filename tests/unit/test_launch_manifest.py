from llm4rec.data.readiness import NO_EXECUTION_FLAG
from llm4rec.experiments.launch_manifest import create_launch_manifest


def test_launch_manifest_contains_no_execution_plan(tmp_path):
    manifest = create_launch_manifest(tmp_path / "launch_manifest.json")
    assert manifest[NO_EXECUTION_FLAG] is True
    assert manifest["api_calls_planned"] == 0
    assert manifest["lora_training_jobs_planned"] == 0
    assert manifest["total_planned_runs"] > 0
    assert (tmp_path / "launch_manifest.json").is_file()
