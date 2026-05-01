import json

from llm4rec.encoders.temporal_graph_encoder import TORCH_AVAILABLE
from llm4rec.trainers.temporal_graph import run_temporal_graph_smoke


def test_temporal_graph_training_smoke_or_clear_skip():
    result = run_temporal_graph_smoke("configs/experiments/phase6_temporal_graph_smoke.yaml")
    metrics = json.loads((result.run_dir / "training_metrics.json").read_text(encoding="utf-8"))
    if TORCH_AVAILABLE:
        assert result.status == "trained"
        assert result.checkpoint_path is not None and result.checkpoint_path.is_file()
    else:
        assert result.status == "skipped_pytorch_unavailable"
        assert metrics["pytorch_available"] is False
