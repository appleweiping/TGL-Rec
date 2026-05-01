import json

from llm4rec.methods.time_graph_evidence import run_time_graph_evidence_smoke


def test_phase6_method_encoder_smoke_outputs_metadata():
    result = run_time_graph_evidence_smoke("configs/experiments/phase6_method_encoder_smoke.yaml")
    run_dir = result.run_dir
    assert (run_dir / "predictions.jsonl").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "metrics.csv").is_file()
    assert (run_dir / "evidence_used.jsonl").is_file()
    first = json.loads((run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["metadata"]["dynamic_encoder_enabled"] is True
    assert "dynamic_encoder_status" in first["metadata"]
    if first["metadata"]["dynamic_encoder_status"]["enabled"]:
        assert first["metadata"]["dynamic_encoder_scores"]
