import json
from pathlib import Path

from llm4rec.evaluation.aggregate import aggregate_multiseed_results
from llm4rec.io.artifacts import write_csv_rows, write_json


def _write_prediction(path: Path, *, user: str, target: str, predicted: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "method": path.parent.name,
        "predicted_items": predicted,
        "scores": [1.0 for _item in predicted],
        "target_item": target,
        "user_id": user,
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _seed_fixture(root: Path, seed: int, *, bm25_recall: float, ours_recall: float) -> Path:
    seed_dir = root if seed == 0 else root / f"seed_{seed}"
    rows = []
    metrics = []
    for method, recall in [("bm25", bm25_recall), ("time_graph_evidence", ours_recall)]:
        pred_path = seed_dir / "tiny" / method / "predictions.jsonl"
        _write_prediction(pred_path, user="u1", target="i1", predicted=["i1", "i2"] if recall else ["i2", "i1"])
        _write_prediction(pred_path, user="u2", target="i3", predicted=["i3", "i4"] if recall else ["i4", "i3"])
        metrics_path = seed_dir / "tiny" / method / "metrics.json"
        write_json(metrics_path, {"overall": {"Recall@5": recall}, "seed": seed})
        rows.append(
            {
                "checkpoint_path": "",
                "dataset": "tiny",
                "failure_reason": "",
                "message": "",
                "method": method,
                "metrics_path": str(metrics_path),
                "predictions_path": str(pred_path),
                "runtime_seconds": 1.0 + seed,
                "status": "succeeded",
            }
        )
        metrics.append(
            {
                "Recall@5": recall,
                "NDCG@5": recall,
                "MRR@10": recall,
                "coverage": recall,
                "dataset": "tiny",
                "long_tail_ratio": 0.0,
                "method": method,
                "novelty": 1.0,
                "runtime_seconds": 1.0 + seed,
                "seed": seed,
            }
        )
    write_csv_rows(seed_dir / "method_status.csv", rows)
    write_csv_rows(seed_dir / "metrics_by_method.csv", metrics)
    checksums = {"tiny": {"candidate_sha256": f"candidate-{seed}", "split_sha256": f"split-{seed}"}}
    write_json(seed_dir / "artifact_checksums_pre.json", checksums)
    write_json(seed_dir / "artifact_checksums_post.json", checksums)
    return seed_dir


def test_phase9c_aggregation_writes_required_outputs(tmp_path: Path):
    seed0 = _seed_fixture(tmp_path / "seed0_source", 0, bm25_recall=0.5, ours_recall=1.0)
    multiseed = tmp_path / "main_accuracy_multiseed"
    _seed_fixture(multiseed, 1, bm25_recall=0.0, ours_recall=1.0)

    result = aggregate_multiseed_results(seed0_dir=seed0, multiseed_dir=multiseed, seeds=[0, 1])

    assert Path(result["aggregate_metrics"]).is_file()
    assert (multiseed / "aggregate_metrics.json").is_file()
    assert (multiseed / "significance_tests.csv").is_file()
    assert (multiseed / "failure_report.json").is_file()
    assert (multiseed / "run_manifest.json").is_file()
    assert (multiseed / "artifact_integrity.json").is_file()
    assert (multiseed / "seed_0" / "linked_or_copied_from_main_accuracy_seed0.json").is_file()
