"""Phase 7 small-scale pilot matrix runner."""

from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

from llm4rec.data.splits import build_user_histories, leave_one_out_split
from llm4rec.evaluation.evaluator import evaluate_predictions
from llm4rec.evaluation.failure_audit import audit_failures
from llm4rec.evaluation.pilot_export import export_pilot_tables
from llm4rec.experiments.config import resolve_experiment_config, resolve_path, save_resolved_config
from llm4rec.experiments.logging import RunLogger
from llm4rec.experiments.resource_estimator import estimate_pilot_resources
from llm4rec.experiments.seeding import set_global_seed
from llm4rec.io.artifacts import ensure_dir, read_jsonl, write_csv_rows, write_json, write_jsonl
from llm4rec.methods.ablation import ablation_switches
from llm4rec.methods.config import load_method_config
from llm4rec.rankers.base import RankingExample
from llm4rec.rankers.bm25 import BM25Ranker
from llm4rec.rankers.mf import MatrixFactorizationRanker
from llm4rec.rankers.popularity import PopularityRanker
from llm4rec.rankers.random import RandomRanker
from llm4rec.rankers.time_graph_evidence_ranker import TimeGraphEvidenceRanker, prediction_row_from_result
from llm4rec.utils.env import collect_environment


def run_pilot_matrix(config_path: str | Path) -> Path:
    """Run a NON_REPORTABLE pilot matrix or pilot ablation config."""

    config = resolve_experiment_config(config_path)
    _assert_pilot_safety(config)
    run_dir = _prepare_run(config)
    logger = RunLogger(run_dir / "logs.txt")
    logger.info("starting Phase 7 pilot")
    resource_path = run_dir / "resource_estimate.json"
    if not resource_path.is_file():
        estimate_pilot_resources(config_path)
    processed_dir = _build_pilot_dataset(config, run_dir / "artifacts" / "processed_dataset")
    if bool(config.get("pilot", {}).get("ablation", False)):
        _run_ablation_pilot(config, run_dir, processed_dir, logger)
    else:
        _run_method_pilot(config, run_dir, processed_dir, logger)
    audit_failures(run_dir)
    export_pilot_tables(run_dir)
    logger.info("Phase 7 pilot completed")
    return run_dir


def _run_method_pilot(config: dict[str, Any], run_dir: Path, processed_dir: Path, logger: RunLogger) -> None:
    method_status: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    for method in config.get("methods", []):
        method_name = str(method.get("name", method)) if isinstance(method, dict) else str(method)
        started = time.perf_counter()
        try:
            predictions, checkpoint = _predictions_for_method(method_name, config, processed_dir, checkpoints_dir)
            predictions_path = ensure_dir(run_dir / "predictions") / f"{method_name}.jsonl"
            write_jsonl(predictions_path, predictions)
            metrics_dir = ensure_dir(run_dir / "metrics" / method_name)
            metrics = _evaluate(config, processed_dir, predictions_path, metrics_dir)
            runtime_ms = (time.perf_counter() - started) * 1000.0
            method_status.append(
                {
                    "NON_REPORTABLE": "NON_REPORTABLE",
                    "checkpoint": checkpoint or "",
                    "message": "",
                    "method": method_name,
                    "runtime_ms": runtime_ms,
                    "status": "succeeded",
                }
            )
            _append_metric_rows(metrics_rows, method_name, metrics)
        except Exception as exc:
            runtime_ms = (time.perf_counter() - started) * 1000.0
            status = "skipped" if "PyTorch" in str(exc) or "torch" in str(exc).lower() else "failed"
            method_status.append(
                {
                    "NON_REPORTABLE": "NON_REPORTABLE",
                    "checkpoint": "",
                    "message": str(exc),
                    "method": method_name,
                    "runtime_ms": runtime_ms,
                    "status": status,
                }
            )
            logger.info(f"method {method_name} {status}: {exc}")
            if not bool(config.get("pilot", {}).get("continue_on_failure", True)):
                raise
    write_csv_rows(run_dir / "method_status.csv", method_status)
    write_csv_rows(run_dir / "metrics_by_method.csv", metrics_rows)
    write_csv_rows(run_dir / "metrics_by_segment.csv", _segment_rows(metrics_rows))
    write_json(
        run_dir / "pilot_manifest.json",
        {
            "NON_REPORTABLE": "NON_REPORTABLE",
            "candidate_protocol": "sampled_fixed",
            "methods": [row["method"] for row in method_status],
            "pilot_reportable": False,
            "split_artifact": str(processed_dir / "interactions.jsonl"),
        },
    )


def _run_ablation_pilot(config: dict[str, Any], run_dir: Path, processed_dir: Path, logger: RunLogger) -> None:
    del logger
    names = [str(name) for name in config.get("ablation", {}).get("names", [])]
    rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    for name in names:
        started = time.perf_counter()
        try:
            method_config = load_method_config("configs/methods/time_graph_evidence.yaml")
            method_config["method"]["name"] = f"time_graph_evidence_rec:{name}"
            method_config["method"]["reportable"] = False
            method_config["ablation"] = ablation_switches(name).to_dict()
            predictions = _time_graph_predictions(config, processed_dir, method_config)
            predictions_path = ensure_dir(run_dir / "predictions") / f"{name}.jsonl"
            write_jsonl(predictions_path, predictions)
            metrics_dir = ensure_dir(run_dir / "metrics" / name)
            metrics = _evaluate(config, processed_dir, predictions_path, metrics_dir)
            overall = dict(metrics.get("overall", {}))
            rows.append(
                {
                    "NON_REPORTABLE": "NON_REPORTABLE",
                    "ablation": name,
                    "message": "",
                    "NDCG@5": overall.get("NDCG@5", 0.0),
                    "Recall@5": overall.get("Recall@5", 0.0),
                    "runtime_ms": (time.perf_counter() - started) * 1000.0,
                    "status": "succeeded",
                }
            )
            manifest.append({"ablation": name, "predictions_path": str(predictions_path), "reportable": False})
            _append_metric_rows(metrics_rows, name, metrics)
        except Exception as exc:
            rows.append(
                {
                    "NON_REPORTABLE": "NON_REPORTABLE",
                    "ablation": name,
                    "message": str(exc),
                    "NDCG@5": 0.0,
                    "Recall@5": 0.0,
                    "runtime_ms": (time.perf_counter() - started) * 1000.0,
                    "status": "failed",
                }
            )
            if not bool(config.get("pilot", {}).get("continue_on_failure", True)):
                raise
    write_csv_rows(run_dir / "ablation_results.csv", rows)
    write_csv_rows(run_dir / "metrics_by_method.csv", metrics_rows)
    write_json(run_dir / "ablation_manifest.json", {"NON_REPORTABLE": "NON_REPORTABLE", "ablations": manifest, "pilot_reportable": False})


def _predictions_for_method(
    method_name: str,
    config: dict[str, Any],
    processed_dir: Path,
    checkpoints_dir: Path,
) -> tuple[list[dict[str, Any]], str | None]:
    train_rows = read_jsonl(processed_dir / "train.jsonl")
    item_rows = read_jsonl(processed_dir / "items.jsonl")
    if method_name == "random":
        ranker = RandomRanker(seed=int(config.get("experiment", {}).get("seed", 0)))
    elif method_name == "popularity":
        ranker = PopularityRanker()
    elif method_name == "bm25":
        ranker = BM25Ranker()
    elif method_name in {"mf", "bpr"}:
        ranker = MatrixFactorizationRanker(epochs=2, factors=4, seed=int(config.get("experiment", {}).get("seed", 0)))
    elif method_name == "time_graph_evidence":
        method_config = load_method_config("configs/methods/time_graph_evidence.yaml")
        method_config["method"]["name"] = "time_graph_evidence_rec"
        method_config["ablation"]["use_dynamic_encoder"] = False
        return _time_graph_predictions(config, processed_dir, method_config), None
    elif method_name == "time_graph_evidence_dynamic":
        method_config = load_method_config("configs/methods/time_graph_evidence.yaml")
        method_config["method"]["name"] = "time_graph_evidence_rec_dynamic"
        method_config["ablation"]["use_dynamic_encoder"] = True
        checkpoint = checkpoints_dir / "temporal_graph_encoder.pt"
        if checkpoint.is_file():
            method_config["encoder"]["checkpoint_path"] = str(checkpoint)
        return _time_graph_predictions(config, processed_dir, method_config), str(checkpoint) if checkpoint.is_file() else None
    elif method_name == "sasrec":
        return _sasrec_pilot_predictions(config, processed_dir, checkpoints_dir)
    elif method_name == "temporal_graph_encoder":
        return _temporal_graph_pilot_predictions(config, processed_dir, checkpoints_dir)
    else:
        raise ValueError(f"unknown pilot method: {method_name}")
    ranker.fit(train_rows, item_rows)
    predictions = _ranker_predictions(method_name, ranker, processed_dir)
    return predictions, None


def _time_graph_predictions(config: dict[str, Any], processed_dir: Path, method_config: dict[str, Any]) -> list[dict[str, Any]]:
    ranker = TimeGraphEvidenceRanker(method_config, candidate_protocol="sampled_fixed")
    ranker.fit(read_jsonl(processed_dir / "train.jsonl"), read_jsonl(processed_dir / "items.jsonl"))
    predictions = []
    for example in _examples(processed_dir):
        result = ranker.rank(example)
        predictions.append(prediction_row_from_result(example=example, result=result, method_name=method_config["method"]["name"], phase="phase7_pilot"))
    return predictions


def _sasrec_pilot_predictions(config: dict[str, Any], processed_dir: Path, checkpoints_dir: Path) -> tuple[list[dict[str, Any]], str]:
    from llm4rec.models.sasrec import SASRecModel, TORCH_AVAILABLE
    from llm4rec.trainers.sasrec import SASRecSequenceDataset, build_item_mappings, left_pad

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch unavailable for SASRec pilot")
    import torch
    from torch.utils.data import DataLoader

    train_rows = read_jsonl(processed_dir / "train.jsonl")
    item_rows = read_jsonl(processed_dir / "items.jsonl")
    item_to_idx, idx_to_item = build_item_mappings(item_rows)
    dataset = SASRecSequenceDataset(train_interactions=train_rows, item_to_idx=item_to_idx, max_seq_len=8, num_negatives=1, seed=0)
    model = SASRecModel(num_items=len(item_to_idx), hidden_dim=16, num_layers=1, num_heads=1, dropout=0.0, max_seq_len=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    if len(dataset) > 0:
        for batch in DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=_collate_sasrec):
            pos = batch["positive"].unsqueeze(1)
            loss = -torch.nn.functional.logsigmoid(model.score_items(batch["input"], pos) - model.score_items(batch["input"], batch["negative"])).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    checkpoint = checkpoints_dir / "sasrec.pt"
    torch.save({"config": {"training": {"hidden_dim": 16, "num_layers": 1, "num_heads": 1, "dropout": 0.0, "max_seq_len": 8}}, "idx_to_item": idx_to_item, "item_to_idx": item_to_idx, "model_state": model.state_dict()}, checkpoint)
    predictions = []
    model.eval()
    for example in _examples(processed_dir):
        seq = torch.tensor([[item_to_idx.get(item, 0) for item in example.history][-8:]], dtype=torch.long)
        seq = torch.nn.functional.pad(seq, (max(0, 8 - seq.size(1)), 0))[:, -8:]
        items = torch.tensor([[item_to_idx.get(item, 0) for item in example.candidate_items]], dtype=torch.long)
        with torch.no_grad():
            scores = model.score_items(seq, items).squeeze(0).tolist()
        predictions.append(_prediction_from_scores(example, "sasrec", scores))
    return predictions, str(checkpoint)


def _temporal_graph_pilot_predictions(config: dict[str, Any], processed_dir: Path, checkpoints_dir: Path) -> tuple[list[dict[str, Any]], str]:
    from llm4rec.encoders.temporal_graph_encoder import TORCH_AVAILABLE, TemporalGraphEncoder, build_temporal_graph_mappings

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch unavailable for TemporalGraphEncoder pilot")
    train_rows = read_jsonl(processed_dir / "train.jsonl")
    item_rows = read_jsonl(processed_dir / "items.jsonl")
    user_to_idx, item_to_idx = build_temporal_graph_mappings(train_rows, item_rows)
    encoder = TemporalGraphEncoder(num_users=len(user_to_idx), num_items=len(item_to_idx), hidden_dim=16, user_to_idx=user_to_idx, item_to_idx=item_to_idx)
    encoder.fit(train_rows)
    checkpoint = checkpoints_dir / "temporal_graph_encoder.pt"
    encoder.save(checkpoint)
    predictions = []
    for example in _examples(processed_dir):
        scores = [encoder.score(example.user_id, item, example.metadata.get("prediction_timestamp")) for item in example.candidate_items]
        predictions.append(_prediction_from_scores(example, "temporal_graph_encoder", scores))
    return predictions, str(checkpoint)


def _ranker_predictions(method_name: str, ranker: Any, processed_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for example in _examples(processed_dir):
        result = ranker.rank(example)
        rows.append(
            {
                "candidate_items": example.candidate_items,
                "domain": example.domain,
                "metadata": {**result.metadata, "NON_REPORTABLE": "NON_REPORTABLE", "phase": "phase7_pilot", "reportable": False},
                "method": method_name,
                "predicted_items": result.items,
                "raw_output": result.raw_output,
                "scores": result.scores,
                "target_item": example.target_item,
                "user_id": example.user_id,
            }
        )
    return rows


def _prediction_from_scores(example: RankingExample, method_name: str, scores: list[float]) -> dict[str, Any]:
    scored = sorted(zip(example.candidate_items, scores), key=lambda pair: (-float(pair[1]), str(pair[0])))
    return {
        "candidate_items": example.candidate_items,
        "domain": example.domain,
        "metadata": {"NON_REPORTABLE": "NON_REPORTABLE", "phase": "phase7_pilot", "reportable": False},
        "method": method_name,
        "predicted_items": [item for item, _score in scored],
        "raw_output": None,
        "scores": [float(score) for _item, score in scored],
        "target_item": example.target_item,
        "user_id": example.user_id,
    }


def _examples(processed_dir: Path) -> list[RankingExample]:
    interactions = read_jsonl(processed_dir / "interactions.jsonl")
    examples = []
    for row in read_jsonl(processed_dir / "candidates.jsonl"):
        if row["split"] != "test":
            continue
        target_ts = _target_timestamp(row, interactions)
        history = [
            str(event["item_id"])
            for event in sorted(interactions, key=lambda event: (float(event.get("timestamp") or -1), str(event["item_id"])))
            if str(event["user_id"]) == str(row["user_id"])
            and str(event.get("split")) != "test"
            and (target_ts is None or event.get("timestamp") is None or float(event["timestamp"]) < target_ts)
        ]
        examples.append(
            RankingExample(
                user_id=str(row["user_id"]),
                history=history,
                target_item=str(row["target_item"]),
                candidate_items=[str(item) for item in row["candidate_items"]],
                domain=row.get("domain"),
                metadata={"prediction_timestamp": target_ts, "split": "test"},
            )
        )
    return examples


def _target_timestamp(candidate_row: dict[str, Any], interactions: list[dict[str, Any]]) -> float | None:
    for event in interactions:
        if str(event["user_id"]) == str(candidate_row["user_id"]) and str(event["item_id"]) == str(candidate_row["target_item"]) and str(event.get("split")) == str(candidate_row["split"]):
            return None if event.get("timestamp") is None else float(event["timestamp"])
    return None


def _build_pilot_dataset(config: dict[str, Any], output_dir: Path) -> Path:
    pilot = dict(config.get("pilot", {}))
    max_users = int(pilot.get("max_users", 200))
    max_items = int(pilot.get("max_items", 1000))
    max_interactions = int(pilot.get("max_interactions", 10000))
    candidate_size = int(pilot.get("candidate_size", 100))
    seed = int(config.get("experiment", {}).get("seed", 0))
    user_count = min(max_users, max(8, max_interactions // 250), 40)
    item_count = min(max_items, max(candidate_size + 5, 120))
    interactions_per_user = max(5, min(12, max_interactions // max(1, user_count)))
    interactions = []
    for user_idx in range(1, user_count + 1):
        for pos in range(interactions_per_user):
            item_idx = ((user_idx * 7 + pos * 11) % item_count) + 1
            interactions.append({"domain": "movielens_sample", "item_id": f"i{item_idx}", "rating": 1.0, "timestamp": float(user_idx * 1000 + pos), "user_id": f"u{user_idx}"})
    labeled = leave_one_out_split(interactions)
    items = [
        {"brand": None, "category": f"genre_{idx % 8}", "description": None, "domain": "movielens_sample", "item_id": f"i{idx}", "raw_text": f"Movie {idx} genre_{idx % 8}", "title": f"Movie {idx}"}
        for idx in range(1, item_count + 1)
    ]
    candidates = _sampled_candidates(labeled, [row["item_id"] for row in items], candidate_size, seed)
    histories = [{"user_id": user_id, "history": history} for user_id, history in sorted(build_user_histories(labeled).items())]
    ensure_dir(output_dir)
    write_jsonl(output_dir / "interactions.jsonl", labeled)
    write_jsonl(output_dir / "train.jsonl", [row for row in labeled if row["split"] == "train"])
    write_jsonl(output_dir / "valid.jsonl", [row for row in labeled if row["split"] == "valid"])
    write_jsonl(output_dir / "test.jsonl", [row for row in labeled if row["split"] == "test"])
    write_jsonl(output_dir / "items.jsonl", items)
    write_jsonl(output_dir / "histories.jsonl", histories)
    write_jsonl(output_dir / "candidates.jsonl", candidates)
    write_json(output_dir / "metadata.json", {"NON_REPORTABLE": "NON_REPORTABLE", "candidate_size": min(candidate_size, item_count), "candidate_protocol": "sampled_fixed", "dataset": "movielens_style_sampled_pilot", "interaction_count": len(interactions), "item_count": item_count, "pilot_reportable": False, "seed": seed, "split_strategy": "leave_one_out", "user_count": user_count})
    return output_dir


def _sampled_candidates(labeled: list[dict[str, Any]], item_ids: list[str], candidate_size: int, seed: int) -> list[dict[str, Any]]:
    catalog = sorted(str(item) for item in item_ids)
    rows = []
    for row in labeled:
        if row["split"] not in {"valid", "test"}:
            continue
        target = str(row["item_id"])
        rng = random.Random(f"{seed}|{row['user_id']}|{target}|{row['split']}")
        negatives = [item for item in catalog if item != target]
        rng.shuffle(negatives)
        candidates = sorted([target, *negatives[: max(0, min(candidate_size, len(catalog)) - 1)]])
        rows.append({"candidate_items": candidates, "domain": row.get("domain"), "split": row["split"], "target_item": target, "user_id": str(row["user_id"])})
    return sorted(rows, key=lambda value: (value["split"], value["user_id"], value["target_item"]))


def _evaluate(config: dict[str, Any], processed_dir: Path, predictions_path: Path, output_dir: Path) -> dict[str, Any]:
    evaluation = dict(config.get("evaluation", {}))
    return evaluate_predictions(predictions_path=predictions_path, item_catalog_path=processed_dir / "items.jsonl", output_dir=output_dir, ks=tuple(int(k) for k in evaluation.get("ks", [1, 5, 10])), candidate_protocol="sampled_fixed")


def _append_metric_rows(rows: list[dict[str, Any]], method: str, metrics: dict[str, Any]) -> None:
    for metric, value in sorted(metrics.get("overall", {}).items()):
        if isinstance(value, (int, float)):
            rows.append({"NON_REPORTABLE": "NON_REPORTABLE", "method": method, "metric": metric, "value": value})


def _segment_rows(metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, "segment": "ALL"} for row in metrics_rows]


def _prepare_run(config: dict[str, Any]) -> Path:
    experiment = dict(config.get("experiment", {}))
    output_root = ensure_dir(resolve_path(experiment.get("output_dir", "outputs/runs")))
    run_dir = output_root / str(experiment.get("run_id", "phase7_pilot"))
    if run_dir.exists() and bool(experiment.get("overwrite", True)):
        _remove_run_dir(run_dir, output_root)
    ensure_dir(run_dir)
    for child in ("predictions", "metrics", "checkpoints", "artifacts"):
        ensure_dir(run_dir / child)
    save_resolved_config(config, run_dir / "resolved_config.yaml")
    write_json(run_dir / "environment.json", collect_environment())
    return run_dir


def _assert_pilot_safety(config: dict[str, Any]) -> None:
    if str(config.get("experiment", {}).get("run_mode")) != "pilot":
        raise ValueError("Phase 7 runner requires run_mode=pilot")
    if bool(config.get("pilot_reportable", False)):
        raise ValueError("pilot configs must set pilot_reportable=false")
    if bool(config.get("llm", {}).get("allow_api_calls", False)):
        raise ValueError("pilot configs must not allow API calls")
    if bool(config.get("training", {}).get("enable_lora_training", False)):
        raise ValueError("pilot configs must not enable LoRA training")


def _remove_run_dir(run_dir: Path, output_root: Path) -> None:
    resolved_run = run_dir.resolve()
    resolved_root = output_root.resolve()
    if resolved_run == resolved_root or resolved_root not in resolved_run.parents:
        raise ValueError(f"Refusing to remove run dir outside output root: {run_dir}")
    shutil.rmtree(resolved_run, ignore_errors=True)


def _collate_sasrec(batch: list[Any]) -> dict[str, Any]:
    import torch

    return {"input": torch.tensor([row.input_indices for row in batch], dtype=torch.long), "positive": torch.tensor([row.positive_index for row in batch], dtype=torch.long), "negative": torch.tensor([row.negative_indices for row in batch], dtype=torch.long)}
