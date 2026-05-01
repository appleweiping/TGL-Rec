import pytest

pytest.importorskip("torch", reason="PyTorch is required for SASRec ranker tests")

from llm4rec.evaluation.prediction_schema import validate_prediction_row
from llm4rec.trainers.sasrec import run_sasrec_smoke


def test_sasrec_smoke_predictions_validate_schema_when_torch_available():
    result = run_sasrec_smoke("configs/experiments/phase6_sasrec_smoke.yaml")
    assert result.status == "trained"
    first = (result.run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0]
    import json

    validate_prediction_row(json.loads(first), candidate_protocol="full_catalog")
