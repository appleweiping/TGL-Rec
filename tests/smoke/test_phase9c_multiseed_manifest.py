import json
from pathlib import Path

from llm4rec.experiments.multiseed_paper import run_multiseed_paper_matrix


def test_phase9c_multiseed_manifest_records_seed0_reuse_policy(tmp_path: Path):
    run_dir = run_multiseed_paper_matrix(
        manifest_path=tmp_path / "missing_manifest.json",
        matrix="main_accuracy",
        seeds=[0],
        datasets=["tiny"],
        methods=["bm25"],
        output_dir=tmp_path / "phase9c",
        candidate_output_mode="compact_ref",
        shared_pool_scoring=True,
        continue_on_failure=True,
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "seed_run_status.json").read_text(encoding="utf-8"))

    assert manifest["protocol_version"] == "protocol_v1"
    assert manifest["seed0_policy"] == "reuse_existing_outputs_only"
    assert status["seeds"][0]["status"] == "skipped"
