import json

from llm4rec.experiments.pilot_runner import run_pilot_matrix


def test_phase7_pilot_matrix_outputs_required_artifacts():
    run_dir = run_pilot_matrix("configs/experiments/phase7_pilot_movielens_sample.yaml")
    for name in [
        "resolved_config.yaml",
        "environment.json",
        "logs.txt",
        "pilot_manifest.json",
        "resource_estimate.json",
        "method_status.csv",
        "metrics_by_method.csv",
        "metrics_by_segment.csv",
        "failure_report.json",
        "pilot_table.csv",
        "pilot_table.tex",
    ]:
        assert (run_dir / name).is_file(), name
    assert "NON_REPORTABLE" in (run_dir / "pilot_table.csv").read_text(encoding="utf-8")
    metadata = json.loads((run_dir / "artifacts" / "processed_dataset" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["candidate_protocol"] == "sampled_fixed"
    assert metadata["pilot_reportable"] is False
