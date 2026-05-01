from llm4rec.experiments.multiseed import run_multiseed


def test_multiseed_smoke_outputs_aggregate_metrics():
    run_dir = run_multiseed("configs/experiments/multiseed_smoke.yaml")
    assert (run_dir / "aggregate_metrics.csv").is_file()
    assert (run_dir / "seed_metrics.csv").is_file()
