from llm4rec.experiments.pilot_runner import run_pilot_matrix


def test_phase7_pilot_ablation_outputs_required_artifacts():
    run_dir = run_pilot_matrix("configs/experiments/phase7_pilot_ablation_sample.yaml")
    for name in [
        "resolved_config.yaml",
        "ablation_manifest.json",
        "ablation_results.csv",
        "ablation_table.csv",
        "failure_report.json",
    ]:
        assert (run_dir / name).is_file(), name
    assert "NON_REPORTABLE" in (run_dir / "ablation_table.csv").read_text(encoding="utf-8")
