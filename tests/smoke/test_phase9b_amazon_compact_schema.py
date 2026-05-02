from __future__ import annotations

import json
import shutil
from pathlib import Path

from llm4rec.experiments.paper_matrix import PaperMatrixRequest, run_paper_matrix
from llm4rec.io.artifacts import write_json, write_jsonl


def test_phase9b_runner_writes_compact_candidate_refs(tmp_path: Path) -> None:
    dataset = "tiny_amazon_compact"
    artifact_dir = tmp_path / "artifacts"
    output_dir = Path("outputs/paper_runs/protocol_v1/test_phase9b_amazon_compact_schema")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    artifact_dir.mkdir(parents=True)
    items_path = tmp_path / "items.jsonl"
    split_path = artifact_dir / "splits.jsonl"
    candidates_path = artifact_dir / "candidates.jsonl"
    pool_path = artifact_dir / "candidate_pool.json"
    config_path = tmp_path / "dataset.yaml"
    experiment_path = tmp_path / "experiment.yaml"
    manifest_path = tmp_path / "manifest.json"
    write_jsonl(
        items_path,
        [
            {"item_id": "i1", "title": "one", "description": None, "category": "a", "brand": None, "domain": "tiny", "raw_text": "one"},
            {"item_id": "i2", "title": "two", "description": None, "category": "a", "brand": None, "domain": "tiny", "raw_text": "two"},
            {"item_id": "i3", "title": "three", "description": None, "category": "b", "brand": None, "domain": "tiny", "raw_text": "three"},
        ],
    )
    write_jsonl(
        split_path,
        [
            {"user_id": "u1", "item_id": "i1", "timestamp": 1, "rating": 1.0, "domain": "tiny", "split": "train"},
            {"user_id": "u1", "item_id": "i3", "timestamp": 2, "rating": 1.0, "domain": "tiny", "split": "test"},
        ],
    )
    write_json(pool_path, {"candidate_items": ["i1", "i2"], "candidate_size": 2, "negative_pool_for_targets_outside_pool": ["i1"]})
    write_jsonl(
        candidates_path,
        [
            {
                "candidate_pool_artifact": str(pool_path),
                "candidate_size": 2,
                "candidate_storage": "shared_pool",
                "split": "test",
                "target_item": "i3",
                "user_id": "u1",
            }
        ],
    )
    items_posix = items_path.as_posix()
    pool_posix = pool_path.as_posix()
    config_posix = config_path.as_posix()
    experiment_posix = experiment_path.as_posix()
    candidates_posix = candidates_path.as_posix()
    split_posix = split_path.as_posix()
    config_path.write_text(
        f"""
dataset:
  name: {dataset}
  adapter: generic_jsonl
  paths:
    items: "{items_posix}"
paper_artifacts:
  protocol_version: protocol_v1
  candidate_protocol: fixed_sampled
  candidate_pool_artifact: "{pool_posix}"
  split_protocol: leave_one_out
""",
        encoding="utf-8",
    )
    experiment_path.write_text(
        f"""
protocol_version: protocol_v1
dataset:
  name: {dataset}
  config_path: "{config_posix}"
""",
        encoding="utf-8",
    )
    write_json(
        manifest_path,
        {
            "api_calls_planned": 0,
            "lora_training_jobs_planned": 0,
            "protocol_version": "protocol_v1",
            "experiments": [
                {
                    "api_calls_allowed": False,
                    "candidate_artifact": candidates_posix,
                    "config_path": experiment_posix,
                    "dataset": dataset,
                    "lora_training_enabled": False,
                    "methods": ["popularity"],
                    "protocol_version": "protocol_v1",
                    "seeds": [0],
                    "split_artifact": split_posix,
                    "split_strategy": "leave_one_out",
                }
            ],
        },
    )

    run_paper_matrix(
        PaperMatrixRequest(
            manifest_path=manifest_path,
            matrix="main_accuracy",
            seed=0,
            datasets=(dataset,),
            methods=("popularity",),
            output_dir=output_dir,
            continue_on_failure=True,
            candidate_output_mode="compact_ref_v1",
            rerun_failed_only=True,
        )
    )

    prediction = json.loads((output_dir / dataset / "popularity" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "candidate_items" not in prediction
    assert prediction["candidate_ref"]["candidate_size"] == 2
    assert prediction["metadata"]["candidate_schema"] == "compact_ref_v1"
