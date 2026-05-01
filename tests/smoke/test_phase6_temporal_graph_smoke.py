import json

from llm4rec.encoders.temporal_graph_encoder import TORCH_AVAILABLE
from llm4rec.trainers.temporal_graph import run_temporal_graph_smoke


def test_phase6_temporal_graph_smoke_or_clear_skip():
    result = run_temporal_graph_smoke("configs/experiments/phase6_temporal_graph_smoke.yaml")
    assert (result.run_dir / "resolved_config.yaml").is_file()
    assert (result.run_dir / "environment.json").is_file()
    assert (result.run_dir / "logs.txt").is_file()
    assert (result.run_dir / "checkpoints").is_dir()
    metrics = json.loads((result.run_dir / "training_metrics.json").read_text(encoding="utf-8"))
    if TORCH_AVAILABLE:
        assert (result.run_dir / "predictions.jsonl").is_file()
        assert (result.run_dir / "metrics.json").is_file()
        assert result.checkpoint_path is not None and result.checkpoint_path.is_file()
    else:
        assert metrics["status"] == "skipped_pytorch_unavailable"
