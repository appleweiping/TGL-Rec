import pytest

pytest.importorskip("torch", reason="PyTorch is required for TemporalGraphRanker tests")

from llm4rec.evaluation.prediction_schema import validate_prediction_row
from llm4rec.trainers.temporal_graph import run_temporal_graph_smoke


def test_temporal_graph_ranker_outputs_prediction_schema_when_torch_available():
    result = run_temporal_graph_smoke("configs/experiments/phase6_temporal_graph_smoke.yaml")
    assert result.status == "trained"
    import json

    first = (result.run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0]
    validate_prediction_row(json.loads(first), candidate_protocol="full_catalog")
