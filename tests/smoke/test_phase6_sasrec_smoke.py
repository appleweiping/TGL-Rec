import json

from llm4rec.models.sasrec import TORCH_AVAILABLE
from llm4rec.trainers.sasrec import run_sasrec_smoke


def test_phase6_sasrec_smoke_or_clear_skip():
    result = run_sasrec_smoke("configs/experiments/phase6_sasrec_smoke.yaml")
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
