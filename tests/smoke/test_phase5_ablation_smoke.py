import json
from pathlib import Path

from llm4rec.methods.ablation import REQUIRED_ABLATIONS
from llm4rec.methods.time_graph_evidence import run_time_graph_evidence_smoke


def test_phase5_ablation_smoke_outputs_required_artifacts():
    result = run_time_graph_evidence_smoke("configs/experiments/phase5_ablation_smoke.yaml")
    run_dir = result.run_dir
    required = [
        "resolved_config.yaml",
        "ablation_results.csv",
        "ablation_manifest.json",
    ]
    for name in required:
        assert (run_dir / name).is_file(), name
    assert (run_dir / "predictions").is_dir()
    assert (run_dir / "metrics").is_dir()
    assert (run_dir / "artifacts").is_dir()
    manifest = json.loads((run_dir / "ablation_manifest.json").read_text(encoding="utf-8"))
    assert [row["ablation"] for row in manifest["ablations"]] == REQUIRED_ABLATIONS


def test_phase5_ablation_outputs_predictions_for_each_ablation():
    predictions_dir = Path("outputs/runs/phase5_ablation_smoke/predictions")
    assert sorted(path.stem for path in predictions_dir.glob("*.jsonl")) == sorted(REQUIRED_ABLATIONS)
