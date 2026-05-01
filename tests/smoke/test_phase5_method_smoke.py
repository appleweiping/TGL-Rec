import json
from pathlib import Path

from llm4rec.evaluation.prediction_schema import validate_prediction_row
from llm4rec.methods.time_graph_evidence import run_time_graph_evidence_smoke


def test_phase5_method_smoke_outputs_required_artifacts():
    result = run_time_graph_evidence_smoke("configs/experiments/phase5_method_smoke.yaml")
    run_dir = result.run_dir
    required = [
        "resolved_config.yaml",
        "environment.json",
        "logs.txt",
        "predictions.jsonl",
        "metrics.json",
        "metrics.csv",
        "evidence_used.jsonl",
        "method_card.md",
    ]
    for name in required:
        assert (run_dir / name).is_file(), name
    first = json.loads((run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    validate_prediction_row(first, candidate_protocol="full_catalog")
    assert first["metadata"]["non_reportable_phase5"] is True
    assert (run_dir / "artifacts").is_dir()


def test_phase5_method_smoke_metrics_are_nonempty():
    metrics = json.loads(Path("outputs/runs/phase5_method_smoke/metrics.json").read_text(encoding="utf-8"))
    assert metrics["num_predictions"] > 0
    assert "Recall@1" in metrics["overall"]
