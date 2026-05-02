"""Protocol-v1 paper-scale accuracy matrix runner.

This module consumes frozen Phase 9A split/candidate artifacts as read-only
inputs. It does not regenerate splits or candidates.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from llm4rec.data.movielens_adapter import load_movielens_style, remap_user_item_ids
from llm4rec.evaluation.export import write_evaluation_outputs
from llm4rec.evaluation.failure_audit import audit_failures
from llm4rec.experiments.config import load_yaml_config, resolve_path, save_resolved_config
from llm4rec.experiments.seeding import set_global_seed
from llm4rec.io.artifacts import ensure_dir, read_jsonl, write_csv_rows, write_json
from llm4rec.metrics.long_tail import long_tail_items
from llm4rec.metrics.novelty import item_novelty
from llm4rec.metrics.ranking import hit_rate_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from llm4rec.models.sasrec import SASRecModel, TORCH_AVAILABLE as SASREC_TORCH_AVAILABLE
from llm4rec.trainers.sasrec import build_item_mappings, left_pad
from llm4rec.utils.env import collect_environment, current_git_commit


REQUIRED_CANDIDATE_CHECKSUMS = {
    ("movielens_full", "candidates"): "89c11b30dcccfc90e20ac5470a2410d60e8c270a37e1673d47ac8ace2128ae4d",
    (
        "amazon_multidomain_filtered_iterative_k3",
        "candidates",
    ): "6be36fe9709dffbcead1867432112b2a0c2a4affbfb03a0b3a23d00a4367c539",
    (
        "amazon_multidomain_filtered_iterative_k3",
        "candidate_pool",
    ): "896278c397bc5c6a33b5805a62b0fd151b8cdbcd26d9c016867d70bc7e3482f6",
}

METHOD_ALIASES = {"mf": "mf_bpr", "bpr": "mf_bpr"}
SUPPORTED_METHODS = {
    "popularity",
    "bm25",
    "mf_bpr",
    "sasrec",
    "temporal_graph_encoder",
    "time_graph_evidence",
    "time_graph_evidence_dynamic",
}
TOP_K = 10
METRIC_KS = (1, 5, 10)


@dataclass(frozen=True)
class PaperMatrixRequest:
    """Resolved CLI request for a paper matrix run."""

    manifest_path: Path
    matrix: str
    seed: int
    datasets: tuple[str, ...]
    methods: tuple[str, ...]
    output_dir: Path
    continue_on_failure: bool


@dataclass
class DatasetBundle:
    """In-memory read-only dataset state built from frozen artifacts."""

    name: str
    config_path: Path
    split_artifact: Path
    candidate_artifact: Path
    candidate_pool_artifact: Path | None
    candidate_protocol: str
    split_strategy: str
    train_rows: list[dict[str, Any]]
    item_rows: list[dict[str, Any]]
    item_catalog: set[str]
    history_by_user: dict[str, list[str]]
    test_timestamp_by_user: dict[str, float | None]
    item_popularity: dict[str, int]
    long_tail: set[str]
    candidate_pool: dict[str, Any] | None
    artifact_checksums: dict[str, Any]


class MetricAccumulator:
    """Streaming metric accumulator for top-k prediction rows."""

    def __init__(
        self,
        *,
        item_catalog: set[str],
        item_popularity: dict[str, int],
        long_tail: set[str],
        candidate_protocol: str,
    ) -> None:
        self.item_catalog = {str(item) for item in item_catalog}
        self.item_popularity = {str(item): int(value) for item, value in item_popularity.items()}
        self.total_popularity = sum(self.item_popularity.values())
        self.long_tail = {str(item) for item in long_tail}
        self.candidate_protocol = candidate_protocol
        self.count = 0
        self.ranking_sums: dict[str, float] = defaultdict(float)
        self.invalid = 0
        self.predicted_total = 0
        self.predicted_set: set[str] = set()
        self.long_tail_hits = 0
        self.novelty_sum = 0.0

    def add(self, row: dict[str, Any]) -> None:
        predicted = [str(item) for item in row.get("predicted_items", [])]
        candidates = [str(item) for item in row.get("candidate_items", [])]
        candidate_set = set(candidates)
        target = str(row["target_item"])
        self.count += 1
        for k in METRIC_KS:
            self.ranking_sums[f"Recall@{k}"] += recall_at_k(predicted, target, k)
            self.ranking_sums[f"HitRate@{k}"] += hit_rate_at_k(predicted, target, k)
            self.ranking_sums[f"NDCG@{k}"] += ndcg_at_k(predicted, target, k)
            self.ranking_sums[f"MRR@{k}"] += mrr_at_k(predicted, target, k)
        for item in predicted[:TOP_K]:
            self.predicted_total += 1
            if item in self.item_catalog:
                self.predicted_set.add(item)
            if item in self.long_tail:
                self.long_tail_hits += 1
            self.novelty_sum += item_novelty(
                item,
                self.item_popularity,
                total_interactions=self.total_popularity,
            )
            if item not in self.item_catalog:
                self.invalid += 1
            elif self.candidate_protocol != "no_candidates" and candidates and item not in candidate_set:
                self.invalid += 1

    def metrics(self) -> dict[str, float]:
        denominator = float(self.count or 1)
        output = {name: value / denominator for name, value in sorted(self.ranking_sums.items())}
        coverage = len(self.predicted_set) / float(len(self.item_catalog) or 1)
        hallucination_rate = self.invalid / float(self.predicted_total or 1)
        output.update(
            {
                "catalog_coverage": coverage,
                "coverage": coverage,
                "hallucination_rate": hallucination_rate,
                "item_coverage": float(len(self.predicted_set)),
                "long_tail_ratio": self.long_tail_hits / float(self.predicted_total or 1),
                "novelty": self.novelty_sum / float(self.predicted_total or 1),
                "num_predictions": float(self.count),
                "validity_rate": 1.0 - hallucination_rate,
            }
        )
        return dict(sorted(output.items()))


class MethodContext:
    """Mutable method state shared between fit and prediction."""

    def __init__(self, method: str, seed: int, run_dir: Path) -> None:
        self.method = method
        self.seed = int(seed)
        self.run_dir = run_dir
        self.checkpoint_path: Path | None = None
        self.training_metrics: dict[str, Any] | None = None
        self.state: dict[str, Any] = {}


def run_paper_matrix(request: PaperMatrixRequest) -> Path:
    """Run the requested protocol-v1 paper matrix."""

    _assert_output_dir(request.output_dir)
    manifest = _load_manifest(request.manifest_path)
    experiments = _experiments_by_dataset(manifest, request.datasets)
    _assert_manifest_safety(manifest, experiments, request)
    output_dir = ensure_dir(resolve_path(request.output_dir))
    _write_root_manifest(output_dir, manifest, request, experiments)

    dataset_bundles = {
        dataset: _load_dataset_bundle(dataset, experiments[dataset], request.seed)
        for dataset in request.datasets
    }
    pre_run_checksums = {
        dataset: bundle.artifact_checksums for dataset, bundle in dataset_bundles.items()
    }
    write_json(output_dir / "artifact_checksums_pre.json", pre_run_checksums)

    method_status: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    for dataset_name in request.datasets:
        bundle = dataset_bundles[dataset_name]
        for raw_method in request.methods:
            method = normalize_method(raw_method)
            started = time.perf_counter()
            method_dir = ensure_dir(output_dir / dataset_name / method)
            resumed = _resume_completed_method(method_dir, dataset_name, method)
            if resumed is not None:
                status_row, metrics = resumed
                method_status.append(status_row)
                metrics_rows.append(_metrics_row(dataset_name, method, metrics))
                write_csv_rows(output_dir / "method_status.csv", method_status)
                write_csv_rows(output_dir / "metrics_by_method.csv", metrics_rows)
                continue
            _prepare_method_dir(method_dir, bundle, method, request, experiments[dataset_name])
            status_row: dict[str, Any]
            try:
                if method not in SUPPORTED_METHODS:
                    raise ValueError(f"unsupported method: {method}")
                result = _run_one_method(bundle, method, method_dir, request)
                runtime = time.perf_counter() - started
                result["metrics"]["overall"]["runtime_seconds"] = runtime
                result["metrics"]["by_method"][method]["runtime_seconds"] = runtime
                result["metrics"]["runtime_seconds"] = runtime
                write_evaluation_outputs(method_dir, result["metrics"])
                write_json(
                    method_dir / "runtime.json",
                    {
                        "runtime_seconds": runtime,
                        "seed": request.seed,
                        "status": "succeeded",
                    },
                )
                status_row = {
                    "checkpoint_path": str(result.get("checkpoint_path") or ""),
                    "dataset": dataset_name,
                    "failure_reason": "",
                    "message": "",
                    "method": method,
                    "metrics_path": str(method_dir / "metrics.json"),
                    "predictions_path": str(method_dir / "predictions.jsonl"),
                    "runtime_seconds": runtime,
                    "status": "succeeded",
                }
                metrics_rows.append(_metrics_row(dataset_name, method, result["metrics"]))
            except Exception as exc:
                runtime = time.perf_counter() - started
                failure = {
                    "dataset": dataset_name,
                    "failure_reason": type(exc).__name__,
                    "message": str(exc),
                    "method": method,
                    "runtime_seconds": runtime,
                    "seed": request.seed,
                    "status": "failed",
                }
                write_json(method_dir / "failure_report.json", failure)
                write_json(method_dir / "runtime.json", failure)
                _append_log(method_dir, f"FAILED {method}: {type(exc).__name__}: {exc}")
                status_row = {
                    "checkpoint_path": "",
                    "dataset": dataset_name,
                    "failure_reason": type(exc).__name__,
                    "message": str(exc),
                    "method": method,
                    "metrics_path": "",
                    "predictions_path": "",
                    "runtime_seconds": runtime,
                    "status": "failed",
                }
                if not request.continue_on_failure:
                    method_status.append(status_row)
                    write_csv_rows(output_dir / "method_status.csv", method_status)
                    raise
            method_status.append(status_row)
            write_csv_rows(output_dir / "method_status.csv", method_status)
            write_csv_rows(output_dir / "metrics_by_method.csv", metrics_rows)

    write_csv_rows(output_dir / "method_status.csv", method_status)
    write_csv_rows(output_dir / "metrics_by_method.csv", metrics_rows)
    write_csv_rows(output_dir / "metrics_by_dataset.csv", _dataset_metric_rows(metrics_rows))
    audit_failures(output_dir)
    export_main_accuracy_tables(output_dir)

    post_run_checksums = {
        dataset: _artifact_checksums(dataset, experiments[dataset])
        for dataset in request.datasets
    }
    write_json(output_dir / "artifact_checksums_post.json", post_run_checksums)
    _assert_artifacts_unchanged(pre_run_checksums, post_run_checksums)
    return output_dir


def normalize_method(method: str) -> str:
    """Normalize method names to output directory names."""

    value = str(method)
    return METHOD_ALIASES.get(value, value)


def export_main_accuracy_tables(run_dir: str | Path) -> dict[str, Any]:
    """Write Phase 9B main-accuracy seed=0 CSV and LaTeX tables."""

    root = Path(run_dir)
    rows = _read_csv(root / "metrics_by_method.csv")
    columns = [
        "dataset",
        "method",
        "status",
        "Recall@5",
        "NDCG@5",
        "MRR@10",
        "coverage",
        "novelty",
        "long_tail_ratio",
        "runtime_seconds",
    ]
    status_by_key = {
        (row.get("dataset", ""), row.get("method", "")): row.get("status", "")
        for row in _read_csv(root / "method_status.csv")
    }
    table_rows: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("dataset", ""), row.get("method", ""))
        table_rows.append({**row, "status": status_by_key.get(key, "succeeded")})
    write_csv_rows(root / "table_main_accuracy_seed0.csv", table_rows, fieldnames=columns)
    (root / "table_main_accuracy_seed0.tex").write_text(
        _latex_table(table_rows, columns),
        encoding="utf-8",
        newline="\n",
    )
    return {
        "table_csv": str(root / "table_main_accuracy_seed0.csv"),
        "table_tex": str(root / "table_main_accuracy_seed0.tex"),
        "row_count": len(table_rows),
    }


def _run_one_method(
    bundle: DatasetBundle,
    method: str,
    method_dir: Path,
    request: PaperMatrixRequest,
) -> dict[str, Any]:
    context = MethodContext(method, request.seed, method_dir)
    _fit_method(bundle, context)
    predictions_path = method_dir / "predictions.jsonl"
    metrics = _write_predictions_and_metrics(bundle, context, predictions_path)
    if context.training_metrics is not None:
        write_json(method_dir / "training_metrics.json", context.training_metrics)
    write_json(
        method_dir / "artifact_checksums.json",
        {
            **bundle.artifact_checksums,
            "verified_before_run": True,
            "verified_required_candidate_checksums": True,
        },
    )
    _append_log(method_dir, f"succeeded method={method} dataset={bundle.name}")
    return {
        "checkpoint_path": context.checkpoint_path,
        "metrics": metrics,
        "predictions_path": predictions_path,
    }


def _fit_method(bundle: DatasetBundle, context: MethodContext) -> None:
    method = context.method
    _resource_guard(bundle, method)
    if method == "popularity":
        context.state["popularity"] = Counter(str(row["item_id"]) for row in bundle.train_rows)
        _write_method_artifact(context.run_dir, "popularity_counts.json", dict(context.state["popularity"]))
    elif method == "bm25":
        context.state["bm25"] = _build_bm25_state(bundle)
        _write_method_artifact(context.run_dir, "bm25_metadata.json", context.state["bm25"]["metadata"])
    elif method == "mf_bpr":
        _fit_bpr(bundle, context)
    elif method == "sasrec":
        _fit_sasrec(bundle, context)
    elif method == "temporal_graph_encoder":
        _fit_temporal_graph_encoder(bundle, context)
    elif method == "time_graph_evidence":
        context.state["evidence"] = _build_evidence_state(bundle)
        _write_method_artifact(context.run_dir, "evidence_metadata.json", context.state["evidence"]["metadata"])
    elif method == "time_graph_evidence_dynamic":
        dynamic_context = MethodContext("temporal_graph_encoder", context.seed, context.run_dir)
        _fit_temporal_graph_encoder(bundle, dynamic_context)
        context.state["temporal_encoder"] = dynamic_context.state["temporal_encoder"]
        context.state["torch"] = dynamic_context.state["torch"]
        context.state["evidence"] = _build_evidence_state(bundle)
        context.checkpoint_path = dynamic_context.checkpoint_path
        context.training_metrics = {
            **(dynamic_context.training_metrics or {}),
            "dynamic_encoder_used_by": "time_graph_evidence_dynamic",
        }
        _write_method_artifact(context.run_dir, "evidence_metadata.json", context.state["evidence"]["metadata"])
    else:
        raise ValueError(f"unsupported method: {method}")


def _write_predictions_and_metrics(
    bundle: DatasetBundle,
    context: MethodContext,
    predictions_path: Path,
) -> dict[str, Any]:
    accumulator = MetricAccumulator(
        item_catalog=bundle.item_catalog,
        item_popularity=bundle.item_popularity,
        long_tail=bundle.long_tail,
        candidate_protocol=bundle.candidate_protocol,
    )
    domain_accumulators: dict[str, MetricAccumulator] = {}
    started = time.perf_counter()
    prediction_count = 0
    candidate_json_cache = _candidate_json_cache(bundle)
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in _iter_candidate_rows(bundle):
            if str(row.get("split")) != "test":
                continue
            candidates, candidate_items_json = _candidate_items_and_json_for_row(
                bundle,
                row,
                candidate_json_cache,
            )
            target = str(row["target_item"])
            if target not in candidates:
                raise ValueError(f"target missing from frozen candidates: dataset={bundle.name} user={row['user_id']}")
            history = [
                item for item in bundle.history_by_user.get(str(row["user_id"]), [])
                if item != target
            ]
            timestamp = bundle.test_timestamp_by_user.get(str(row["user_id"]))
            top_items, top_scores, metadata = _rank_top_k(
                bundle,
                context,
                user_id=str(row["user_id"]),
                history=history,
                target_item=target,
                candidates=candidates,
                prediction_timestamp=timestamp,
            )
            prediction = {
                "candidate_items": candidates,
                "domain": row.get("domain"),
                "metadata": {
                    **metadata,
                    "candidate_artifact": str(bundle.candidate_artifact),
                    "history_source": "train_split_only",
                    "protocol_version": "protocol_v1",
                    "ranked_cutoff": TOP_K,
                    "seed": context.seed,
                    "split": "test",
                },
                "method": context.method,
                "predicted_items": top_items,
                "raw_output": None,
                "scores": top_scores,
                "target_item": target,
                "user_id": str(row["user_id"]),
            }
            _write_prediction_row(handle, prediction, candidate_items_json)
            accumulator.add(prediction)
            domain = str(row.get("domain") or "unknown")
            domain_accumulators.setdefault(
                domain,
                MetricAccumulator(
                    item_catalog=bundle.item_catalog,
                    item_popularity=bundle.item_popularity,
                    long_tail=bundle.long_tail,
                    candidate_protocol=bundle.candidate_protocol,
                ),
            ).add(prediction)
            prediction_count += 1
            if prediction_count % 50000 == 0:
                _append_log(
                    context.run_dir,
                    f"predictions_written={prediction_count} elapsed_seconds={time.perf_counter() - started:.2f}",
                )
    overall = accumulator.metrics()
    by_domain = {domain: acc.metrics() for domain, acc in sorted(domain_accumulators.items())}
    metrics = {
        "artifact_checksums": bundle.artifact_checksums,
        "by_domain": by_domain,
        "by_method": {context.method: dict(overall)},
        "candidate_protocol": bundle.candidate_protocol,
        "dataset": bundle.name,
        "matrix": "main_accuracy",
        "num_predictions": prediction_count,
        "overall": overall,
        "protocol_version": "protocol_v1",
        "seed": context.seed,
        "split_strategy": bundle.split_strategy,
    }
    return metrics


def _rank_top_k(
    bundle: DatasetBundle,
    context: MethodContext,
    *,
    user_id: str,
    history: list[str],
    target_item: str,
    candidates: list[str],
    prediction_timestamp: float | None,
) -> tuple[list[str], list[float], dict[str, Any]]:
    method = context.method
    if method == "popularity":
        counts: Counter[str] = context.state["popularity"]
        scores = [(float(counts.get(item, 0)), item) for item in candidates]
        return _top_k_from_pairs(scores), _top_k_scores(scores), {"fit_on": "train_split_only"}
    if method == "bm25":
        scores = _bm25_scores(context.state["bm25"], history, candidates)
        return _top_k_from_pairs(scores), _top_k_scores(scores), {"scorer": "local_bm25"}
    if method == "mf_bpr":
        scores = _bpr_scores(context, user_id, candidates)
        return _top_k_from_pairs(scores), _top_k_scores(scores), {"loss": "bpr", "trained_on": "train_split_only"}
    if method == "sasrec":
        scores = _sasrec_scores(context, history, candidates)
        return _top_k_from_pairs(scores), _top_k_scores(scores), {"sequential_baseline": "sasrec"}
    if method == "temporal_graph_encoder":
        scores = _temporal_encoder_scores(context, user_id, candidates, prediction_timestamp)
        return _top_k_from_pairs(scores), _top_k_scores(scores), {"dynamic_encoder": "temporal_graph_encoder"}
    if method == "time_graph_evidence":
        scores = _evidence_scores(context.state["evidence"], history, candidates)
        return _top_k_from_pairs(scores), _top_k_scores(scores), {"evidence_constructed_from": "train_only"}
    if method == "time_graph_evidence_dynamic":
        evidence_scores = dict(_evidence_scores(context.state["evidence"], history, candidates))
        dynamic_scores = dict(_temporal_encoder_scores(context, user_id, candidates, prediction_timestamp))
        scores = [
            (float(evidence_scores.get(item, 0.0)) + 0.25 * float(dynamic_scores.get(item, 0.0)), item)
            for item in candidates
        ]
        return (
            _top_k_from_pairs(scores),
            _top_k_scores(scores),
            {
                "dynamic_encoder": "temporal_graph_encoder",
                "dynamic_encoder_score_metadata": "weighted_additive",
                "evidence_constructed_from": "train_only",
            },
        )
    raise ValueError(f"unsupported method: {method}")


def _top_k_from_pairs(score_item_pairs: list[tuple[float, str]]) -> list[str]:
    top = heapq.nsmallest(
        TOP_K,
        ((-float(score), str(item)) for score, item in score_item_pairs),
    )
    return [item for _neg_score, item in top]


def _top_k_scores(score_item_pairs: list[tuple[float, str]]) -> list[float]:
    top = heapq.nsmallest(
        TOP_K,
        ((-float(score), str(item)) for score, item in score_item_pairs),
    )
    return [-float(neg_score) for neg_score, _item in top]


def _build_bm25_state(bundle: DatasetBundle) -> dict[str, Any]:
    import re

    token_re = re.compile(r"[a-z0-9]+")
    item_texts = {str(row["item_id"]): _item_text(row) for row in bundle.item_rows}
    candidate_scoring_items = _candidate_scoring_item_ids(bundle)
    document_frequency: Counter[str] = Counter()
    doc_lengths: dict[str, int] = {}
    selected_tf: dict[str, Counter[str]] = {}
    total_len = 0
    for item_id, text in item_texts.items():
        tokens = token_re.findall(text.lower())
        total_len += len(tokens)
        doc_lengths[item_id] = len(tokens)
        unique = set(tokens)
        document_frequency.update(unique)
        if item_id in candidate_scoring_items:
            selected_tf[item_id] = Counter(tokens)
    num_documents = len(item_texts)
    average_doc_length = total_len / float(num_documents or 1)
    inverted: dict[str, list[tuple[str, float]]] = defaultdict(list)
    k1 = 1.5
    b = 0.75
    for item_id, tf in selected_tf.items():
        doc_len = doc_lengths.get(item_id, 0)
        if doc_len <= 0:
            continue
        for token, freq in tf.items():
            df = document_frequency.get(token, 0)
            idf = math.log(1.0 + (num_documents - df + 0.5) / (df + 0.5))
            denom = freq + k1 * (1.0 - b + b * doc_len / (average_doc_length or 1.0))
            contribution = idf * (freq * (k1 + 1.0)) / denom
            inverted[token].append((item_id, float(contribution)))
    return {
        "candidate_scoring_items": candidate_scoring_items,
        "inverted": dict(inverted),
        "item_texts": item_texts,
        "metadata": {
            "average_doc_length": average_doc_length,
            "b": b,
            "candidate_scoring_items": len(candidate_scoring_items),
            "k1": k1,
            "num_documents": num_documents,
            "vocabulary_size": len(document_frequency),
        },
        "token_re": token_re,
    }


def _bm25_scores(state: dict[str, Any], history: list[str], candidates: list[str]) -> list[tuple[float, str]]:
    query_tokens: list[str] = []
    token_re = state["token_re"]
    item_texts: dict[str, str] = state["item_texts"]
    for item in history:
        query_tokens.extend(token_re.findall(item_texts.get(str(item), "").lower()))
    query_counts = Counter(query_tokens)
    candidate_set = {str(item) for item in candidates}
    scores: dict[str, float] = {str(item): 0.0 for item in candidates}
    for token, count in query_counts.items():
        for item_id, contribution in state["inverted"].get(token, []):
            if item_id in candidate_set:
                scores[item_id] += float(count) * float(contribution)
    return [(score, item) for item, score in scores.items()]


def _fit_bpr(bundle: DatasetBundle, context: MethodContext) -> None:
    torch = _require_torch("MF/BPR")
    set_global_seed(context.seed)
    user_to_idx = {user: idx for idx, user in enumerate(sorted({str(row["user_id"]) for row in bundle.train_rows}), start=1)}
    item_to_idx = {item: idx for idx, item in enumerate(sorted(bundle.item_catalog), start=1)}
    pairs = [
        (user_to_idx[str(row["user_id"])], item_to_idx[str(row["item_id"])])
        for row in bundle.train_rows
        if str(row["user_id"]) in user_to_idx and str(row["item_id"]) in item_to_idx
    ]
    device = torch.device("cpu")
    factors = 32
    batch_size = 8192
    user_emb = torch.nn.Embedding(len(user_to_idx) + 1, factors, padding_idx=0).to(device)
    item_emb = torch.nn.Embedding(len(item_to_idx) + 1, factors, padding_idx=0).to(device)
    torch.nn.init.normal_(user_emb.weight, mean=0.0, std=0.02)
    torch.nn.init.normal_(item_emb.weight, mean=0.0, std=0.02)
    optimizer = torch.optim.AdamW([*user_emb.parameters(), *item_emb.parameters()], lr=0.01, weight_decay=0.0001)
    generator = torch.Generator(device=device).manual_seed(context.seed)
    order = torch.randperm(len(pairs), generator=generator)
    user_tensor = torch.tensor([pair[0] for pair in pairs], dtype=torch.long, device=device)
    pos_tensor = torch.tensor([pair[1] for pair in pairs], dtype=torch.long, device=device)
    losses: list[float] = []
    user_emb.train()
    item_emb.train()
    for start in range(0, len(pairs), batch_size):
        idx = order[start : start + batch_size]
        users = user_tensor[idx]
        positives = pos_tensor[idx]
        negatives = torch.randint(1, len(item_to_idx) + 1, positives.shape, generator=generator, device=device)
        pos_scores = (user_emb(users) * item_emb(positives)).sum(dim=-1)
        neg_scores = (user_emb(users) * item_emb(negatives)).sum(dim=-1)
        loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    checkpoint_dir = ensure_dir(context.run_dir / "checkpoints")
    checkpoint = checkpoint_dir / "mf_bpr.pt"
    torch.save(
        {
            "factors": factors,
            "item_state": item_emb.state_dict(),
            "item_to_idx": item_to_idx,
            "seed": context.seed,
            "user_state": user_emb.state_dict(),
            "user_to_idx": user_to_idx,
        },
        checkpoint,
    )
    context.checkpoint_path = checkpoint
    context.state.update({"item_emb": item_emb.eval(), "item_to_idx": item_to_idx, "torch": torch, "user_emb": user_emb.eval(), "user_to_idx": user_to_idx})
    context.training_metrics = {
        "batch_size": batch_size,
        "checkpoint_path": str(checkpoint),
        "epochs": 1,
        "factors": factors,
        "final_loss": losses[-1] if losses else None,
        "loss_count": len(losses),
        "pytorch_available": True,
        "seed": context.seed,
        "status": "trained",
        "training_examples": len(pairs),
    }


def _bpr_scores(context: MethodContext, user_id: str, candidates: list[str]) -> list[tuple[float, str]]:
    torch = context.state["torch"]
    user_idx = context.state["user_to_idx"].get(str(user_id), 0)
    candidate_indices = [context.state["item_to_idx"].get(str(item), 0) for item in candidates]
    with torch.no_grad():
        user = context.state["user_emb"](torch.tensor([user_idx], dtype=torch.long))
        items = context.state["item_emb"](torch.tensor(candidate_indices, dtype=torch.long))
        scores = (user * items).sum(dim=-1).detach().cpu().tolist()
    return [(float(score), str(item)) for score, item in zip(scores, candidates)]


def _fit_sasrec(bundle: DatasetBundle, context: MethodContext) -> None:
    torch = _require_torch("SASRec")
    if not SASREC_TORCH_AVAILABLE:
        raise RuntimeError("PyTorch unavailable for SASRec")
    set_global_seed(context.seed)
    item_to_idx, idx_to_item = build_item_mappings(bundle.item_rows)
    max_seq_len = 20
    hidden_dim = 32
    model = SASRecModel(
        num_items=len(item_to_idx),
        hidden_dim=hidden_dim,
        num_layers=1,
        num_heads=1,
        dropout=0.0,
        max_seq_len=max_seq_len,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    sequences = _train_sequences(bundle.train_rows)
    examples = _sasrec_training_examples(sequences, item_to_idx, max_seq_len)
    generator = torch.Generator().manual_seed(context.seed)
    batch_size = 512
    losses: list[float] = []
    model.train()
    for batch in _batched(examples, batch_size):
        inputs = torch.tensor([row[0] for row in batch], dtype=torch.long)
        positives = torch.tensor([[row[1]] for row in batch], dtype=torch.long)
        negatives = torch.randint(1, len(item_to_idx) + 1, positives.shape, generator=generator)
        pos_scores = model.score_items(inputs, positives)
        neg_scores = model.score_items(inputs, negatives)
        loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    checkpoint_dir = ensure_dir(context.run_dir / "checkpoints")
    checkpoint = checkpoint_dir / "sasrec.pt"
    torch.save(
        {
            "config": {
                "training": {
                    "dropout": 0.0,
                    "hidden_dim": hidden_dim,
                    "max_seq_len": max_seq_len,
                    "num_heads": 1,
                    "num_layers": 1,
                    "seed": context.seed,
                }
            },
            "idx_to_item": idx_to_item,
            "item_to_idx": item_to_idx,
            "model_state": model.state_dict(),
        },
        checkpoint,
    )
    context.checkpoint_path = checkpoint
    context.state.update({"item_to_idx": item_to_idx, "max_seq_len": max_seq_len, "model": model.eval(), "torch": torch})
    context.training_metrics = {
        "batch_size": batch_size,
        "checkpoint_path": str(checkpoint),
        "epochs": 1,
        "final_loss": losses[-1] if losses else None,
        "hidden_dim": hidden_dim,
        "loss_count": len(losses),
        "max_seq_len": max_seq_len,
        "pytorch_available": True,
        "seed": context.seed,
        "status": "trained",
        "training_examples": len(examples),
    }


def _sasrec_scores(context: MethodContext, history: list[str], candidates: list[str]) -> list[tuple[float, str]]:
    torch = context.state["torch"]
    item_to_idx = context.state["item_to_idx"]
    max_seq_len = int(context.state["max_seq_len"])
    sequence = [item_to_idx.get(str(item), 0) for item in history]
    candidate_indices = [item_to_idx.get(str(item), 0) for item in candidates]
    with torch.no_grad():
        seq_tensor = torch.tensor([left_pad(sequence, max_seq_len)], dtype=torch.long)
        item_tensor = torch.tensor([candidate_indices], dtype=torch.long)
        scores = context.state["model"].score_items(seq_tensor, item_tensor).squeeze(0).detach().cpu().tolist()
    return [(float(score), str(item)) for score, item in zip(scores, candidates)]


def _fit_temporal_graph_encoder(bundle: DatasetBundle, context: MethodContext) -> None:
    torch = _require_torch("TemporalGraphEncoder")
    from llm4rec.encoders.temporal_graph_encoder import TemporalGraphEncoder, build_temporal_graph_mappings

    set_global_seed(context.seed)
    user_to_idx, item_to_idx = build_temporal_graph_mappings(bundle.train_rows, bundle.item_rows)
    encoder = TemporalGraphEncoder(
        num_users=len(user_to_idx),
        num_items=len(item_to_idx),
        hidden_dim=32,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
    )
    encoder.fit(bundle.train_rows)
    checkpoint_dir = ensure_dir(context.run_dir / "checkpoints")
    checkpoint = checkpoint_dir / "temporal_graph_encoder.pt"
    encoder.save(checkpoint)
    context.checkpoint_path = checkpoint
    context.state.update({"temporal_encoder": encoder.eval(), "torch": torch})
    context.training_metrics = {
        "checkpoint_path": str(checkpoint),
        "constructed_from": "train_split_only",
        "events": len(bundle.train_rows),
        "hidden_dim": 32,
        "pytorch_available": True,
        "seed": context.seed,
        "status": "trained",
    }


def _temporal_encoder_scores(
    context: MethodContext,
    user_id: str,
    candidates: list[str],
    timestamp: float | None,
) -> list[tuple[float, str]]:
    encoder = context.state["temporal_encoder"]
    torch = context.state["torch"]
    user_idx = encoder.user_to_idx.get(str(user_id), 0)
    item_indices = [encoder.item_to_idx.get(str(item), 0) for item in candidates]
    with torch.no_grad():
        user_tensor = torch.tensor([user_idx], dtype=torch.long)
        item_tensor = torch.tensor(item_indices, dtype=torch.long)
        time_value = torch.tensor([[0.0 if timestamp is None else float(timestamp)]], dtype=torch.float32)
        time_scale = torch.log1p(time_value.clamp(min=0.0))
        user_vec = encoder.user_memory(user_tensor) + encoder.time_projection(time_scale)
        item_vec = encoder.item_memory(item_tensor)
        scores = (user_vec * item_vec).sum(dim=-1).detach().cpu().tolist()
    return [(score, str(item)) for score, item in zip(scores, candidates)]


def _build_evidence_state(bundle: DatasetBundle) -> dict[str, Any]:
    by_user = _train_sequences(bundle.train_rows)
    transition_counts: Counter[tuple[str, str]] = Counter()
    for items in by_user.values():
        for left, right in zip(items, items[1:]):
            transition_counts[(left, right)] += 1
    category_by_item = {str(row["item_id"]): str(row.get("category") or "") for row in bundle.item_rows}
    return {
        "category_by_item": category_by_item,
        "metadata": {
            "constructed_from": "train_split_only",
            "transition_edges": len(transition_counts),
            "use_dynamic_encoder": False,
        },
        "transition_counts": transition_counts,
    }


def _evidence_scores(state: dict[str, Any], history: list[str], candidates: list[str]) -> list[tuple[float, str]]:
    transition_counts: Counter[tuple[str, str]] = state["transition_counts"]
    category_by_item: dict[str, str] = state["category_by_item"]
    recent = list(reversed(history[-3:]))
    recent_categories = [category_by_item.get(item, "") for item in recent if category_by_item.get(item, "")]
    dominant_category = Counter(recent_categories).most_common(1)[0][0] if recent_categories else ""
    scores: list[tuple[float, str]] = []
    for candidate in candidates:
        score = 0.0
        for rank, source in enumerate(recent, start=1):
            score += float(transition_counts.get((source, candidate), 0))
            if source != candidate:
                score += 0.05 / float(rank)
        if dominant_category and category_by_item.get(candidate, "") == dominant_category:
            score += 0.1
        scores.append((score, str(candidate)))
    return scores


def _load_dataset_bundle(dataset_name: str, experiment: dict[str, Any], seed: int) -> DatasetBundle:
    split_artifact = resolve_path(experiment["split_artifact"])
    candidate_artifact = resolve_path(experiment["candidate_artifact"])
    config_path = resolve_path(experiment["config_path"])
    experiment_config = load_yaml_config(config_path)
    dataset_config_path = resolve_path(experiment_config["dataset"]["config_path"])
    dataset_config = load_yaml_config(dataset_config_path)
    item_rows = _load_item_rows(dataset_config)
    train_rows, history_by_user, test_timestamp_by_user = _load_split_state(split_artifact)
    item_catalog = {str(row["item_id"]) for row in item_rows}
    item_popularity = Counter(str(row["item_id"]) for row in train_rows)
    for item in item_catalog:
        item_popularity.setdefault(item, 0)
    candidate_pool_artifact = _candidate_pool_path(dataset_config, experiment)
    candidate_pool = None
    if candidate_pool_artifact is not None and candidate_pool_artifact.is_file():
        candidate_pool = json.loads(candidate_pool_artifact.read_text(encoding="utf-8"))
    checksums = _artifact_checksums(dataset_name, experiment, candidate_pool_artifact=candidate_pool_artifact)
    _verify_required_checksums(dataset_name, checksums)
    return DatasetBundle(
        name=dataset_name,
        config_path=dataset_config_path,
        split_artifact=split_artifact,
        candidate_artifact=candidate_artifact,
        candidate_pool_artifact=candidate_pool_artifact,
        candidate_protocol=str(dataset_config.get("paper_artifacts", {}).get("candidate_protocol", experiment.get("candidate_strategy", "fixed_shared_candidates"))),
        split_strategy=str(dataset_config.get("paper_artifacts", {}).get("split_protocol", experiment.get("split_strategy", "leave_one_out"))),
        train_rows=train_rows,
        item_rows=item_rows,
        item_catalog=item_catalog,
        history_by_user=history_by_user,
        test_timestamp_by_user=test_timestamp_by_user,
        item_popularity=dict(item_popularity),
        long_tail=long_tail_items(dict(item_popularity), quantile=0.2),
        candidate_pool=candidate_pool,
        artifact_checksums=checksums,
    )


def _load_item_rows(dataset_config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = dict(dataset_config.get("dataset", dataset_config))
    adapter = str(dataset.get("adapter", "generic_jsonl"))
    paths = dict(dataset.get("paths", {}))
    if adapter == "movielens_style":
        interactions, items = load_movielens_style(paths)
        item_ids_in_data = {str(row["item_id"]) for row in interactions}
        items = [row for row in items if str(row["item_id"]) in item_ids_in_data]
        _interactions, remapped_items = remap_user_item_ids(interactions, items)
        return remapped_items
    items_path = paths.get("items")
    if not items_path:
        raise FileNotFoundError("dataset config missing paths.items")
    return read_jsonl(resolve_path(items_path))


def _load_split_state(
    split_artifact: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, float | None]]:
    train_rows: list[dict[str, Any]] = []
    by_user_train: dict[str, list[dict[str, Any]]] = defaultdict(list)
    test_timestamp_by_user: dict[str, float | None] = {}
    with split_artifact.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            split = str(row.get("split"))
            user = str(row["user_id"])
            if split == "train":
                train_rows.append(row)
                by_user_train[user].append(row)
            elif split == "test":
                timestamp = row.get("timestamp")
                test_timestamp_by_user[user] = None if timestamp is None else float(timestamp)
    history_by_user = {
        user: [
            str(row["item_id"])
            for row in sorted(
                rows,
                key=lambda value: (
                    float(value["timestamp"]) if value.get("timestamp") is not None else -1.0,
                    str(value["item_id"]),
                ),
            )
        ]
        for user, rows in by_user_train.items()
    }
    return train_rows, history_by_user, test_timestamp_by_user


def _iter_candidate_rows(bundle: DatasetBundle) -> Iterable[dict[str, Any]]:
    with bundle.candidate_artifact.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("split")) == "test":
                yield row


def _candidate_items_for_row(bundle: DatasetBundle, row: dict[str, Any]) -> list[str]:
    if row.get("candidate_storage") == "shared_pool":
        if not bundle.candidate_pool:
            raise FileNotFoundError(f"missing candidate pool for {bundle.name}")
        target = str(row["target_item"])
        pool = [str(item) for item in bundle.candidate_pool["candidate_items"]]
        if target in pool:
            return pool
        negatives = [str(item) for item in bundle.candidate_pool.get("negative_pool_for_targets_outside_pool", pool[:-1])]
        candidates = [*negatives, target]
        expected_size = int(row.get("candidate_size", len(pool)))
        if len(candidates) != expected_size:
            raise ValueError(f"expanded shared-pool candidate size mismatch: expected={expected_size} actual={len(candidates)}")
        return candidates
    return [str(item) for item in row.get("candidate_items", [])]


def _candidate_items_and_json_for_row(
    bundle: DatasetBundle,
    row: dict[str, Any],
    cache: dict[str, Any],
) -> tuple[list[str], str]:
    if row.get("candidate_storage") != "shared_pool":
        candidates = _candidate_items_for_row(bundle, row)
        return candidates, json.dumps(candidates, ensure_ascii=True, separators=(",", ":"))
    if not bundle.candidate_pool:
        raise FileNotFoundError(f"missing candidate pool for {bundle.name}")
    target = str(row["target_item"])
    pool = cache["pool"]
    if target in cache["pool_set"]:
        return pool, cache["pool_json"]
    negatives = cache["negative_pool"]
    candidates = [*negatives, target]
    expected_size = int(row.get("candidate_size", len(pool)))
    if len(candidates) != expected_size:
        raise ValueError(f"expanded shared-pool candidate size mismatch: expected={expected_size} actual={len(candidates)}")
    return candidates, f"{cache['negative_pool_json_prefix']},{json.dumps(target, ensure_ascii=True)}]"


def _candidate_json_cache(bundle: DatasetBundle) -> dict[str, Any]:
    if not bundle.candidate_pool:
        return {}
    pool = [str(item) for item in bundle.candidate_pool["candidate_items"]]
    negatives = [
        str(item)
        for item in bundle.candidate_pool.get("negative_pool_for_targets_outside_pool", pool[:-1])
    ]
    negative_json = json.dumps(negatives, ensure_ascii=True, separators=(",", ":"))
    return {
        "negative_pool": negatives,
        "negative_pool_json_prefix": negative_json[:-1],
        "pool": pool,
        "pool_json": json.dumps(pool, ensure_ascii=True, separators=(",", ":")),
        "pool_set": set(pool),
    }


def _write_prediction_row(handle: Any, prediction: dict[str, Any], candidate_items_json: str) -> None:
    row = dict(prediction)
    row.pop("candidate_items", None)
    text = json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    handle.write('{"candidate_items":' + candidate_items_json + "," + text[1:] + "\n")


def _candidate_scoring_item_ids(bundle: DatasetBundle) -> set[str]:
    items: set[str] = set()
    if bundle.candidate_pool:
        items.update(str(item) for item in bundle.candidate_pool.get("candidate_items", []))
    with bundle.candidate_artifact.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("split")) != "test":
                continue
            items.add(str(row["target_item"]))
            if row.get("candidate_items"):
                items.update(str(item) for item in row["candidate_items"])
    return items


def _artifact_checksums(
    dataset_name: str,
    experiment: dict[str, Any],
    *,
    candidate_pool_artifact: Path | None = None,
) -> dict[str, Any]:
    split = resolve_path(experiment["split_artifact"])
    candidates = resolve_path(experiment["candidate_artifact"])
    output: dict[str, Any] = {
        "candidate_artifact": str(candidates),
        "candidate_sha256": sha256_file(candidates),
        "dataset": dataset_name,
        "split_artifact": str(split),
        "split_sha256": sha256_file(split),
    }
    pool = candidate_pool_artifact
    if pool is None:
        pool = _candidate_pool_from_candidate_path(candidates)
    if pool is not None and pool.is_file():
        output["candidate_pool_artifact"] = str(pool)
        output["candidate_pool_sha256"] = sha256_file(pool)
    return output


def _verify_required_checksums(dataset_name: str, checksums: dict[str, Any]) -> None:
    candidate_key = (dataset_name, "candidates")
    if candidate_key in REQUIRED_CANDIDATE_CHECKSUMS:
        expected = REQUIRED_CANDIDATE_CHECKSUMS[candidate_key]
        actual = str(checksums.get("candidate_sha256", "")).lower()
        if actual != expected:
            raise ValueError(f"{dataset_name} candidate checksum mismatch: expected={expected} actual={actual}")
    pool_key = (dataset_name, "candidate_pool")
    if pool_key in REQUIRED_CANDIDATE_CHECKSUMS:
        expected = REQUIRED_CANDIDATE_CHECKSUMS[pool_key]
        actual = str(checksums.get("candidate_pool_sha256", "")).lower()
        if actual != expected:
            raise ValueError(f"{dataset_name} candidate pool checksum mismatch: expected={expected} actual={actual}")


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_pool_path(dataset_config: dict[str, Any], experiment: dict[str, Any]) -> Path | None:
    paper = dict(dataset_config.get("paper_artifacts", {}))
    value = paper.get("candidate_pool_artifact") or experiment.get("candidate_pool_artifact")
    if value:
        return resolve_path(value)
    return _candidate_pool_from_candidate_path(resolve_path(experiment["candidate_artifact"]))


def _candidate_pool_from_candidate_path(candidate_artifact: Path) -> Path | None:
    pool = candidate_artifact.parent / "candidate_pool.json"
    return pool if pool.is_file() else None


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest_path = resolve_path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing launch manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _experiments_by_dataset(manifest: dict[str, Any], datasets: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for experiment in manifest.get("experiments", []):
        if str(experiment.get("dataset")) not in datasets:
            continue
        methods = {normalize_method(str(method)) for method in experiment.get("methods", [])}
        if "popularity" not in methods:
            continue
        output[str(experiment["dataset"])] = dict(experiment)
    missing = sorted(set(datasets) - set(output))
    if missing:
        raise ValueError(f"manifest missing requested datasets: {missing}")
    return output


def _assert_manifest_safety(
    manifest: dict[str, Any],
    experiments: dict[str, dict[str, Any]],
    request: PaperMatrixRequest,
) -> None:
    if str(manifest.get("protocol_version")) != "protocol_v1":
        raise ValueError("Phase 9B requires protocol_version=protocol_v1")
    if request.seed != 0:
        raise ValueError("Phase 9B is seed=0 only")
    if request.matrix != "main_accuracy":
        raise ValueError("Phase 9B runner only supports matrix=main_accuracy")
    if int(manifest.get("api_calls_planned", 0)) != 0:
        raise ValueError("manifest plans API calls")
    if int(manifest.get("lora_training_jobs_planned", 0)) != 0:
        raise ValueError("manifest plans LoRA jobs")
    for dataset, experiment in experiments.items():
        if str(experiment.get("protocol_version")) != "protocol_v1":
            raise ValueError(f"{dataset} experiment is not protocol_v1")
        if bool(experiment.get("api_calls_allowed", True)):
            raise ValueError(f"{dataset} allows API calls")
        if bool(experiment.get("lora_training_enabled", True)):
            raise ValueError(f"{dataset} enables LoRA training")
        if request.seed not in {int(seed) for seed in experiment.get("seeds", [])}:
            raise ValueError(f"{dataset} does not include seed={request.seed}")


def _assert_output_dir(output_dir: Path) -> None:
    resolved = resolve_path(output_dir)
    allowed = resolve_path("outputs/paper_runs/protocol_v1")
    if resolved != allowed and allowed not in resolved.parents:
        raise ValueError(f"Phase 9B output_dir must be under {allowed}: {resolved}")


def _assert_artifacts_unchanged(
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    if before != after:
        raise ValueError("frozen artifact checksum changed during Phase 9B run")


def _write_root_manifest(
    output_dir: Path,
    manifest: dict[str, Any],
    request: PaperMatrixRequest,
    experiments: dict[str, dict[str, Any]],
) -> None:
    write_json(
        output_dir / "run_manifest.json",
        {
            "api_calls_allowed": False,
            "code_commit": current_git_commit(resolve_path(".")),
            "continue_on_failure": request.continue_on_failure,
            "datasets": list(request.datasets),
            "experiments": experiments,
            "launch_manifest": str(resolve_path(request.manifest_path)),
            "launch_status_before_run": manifest.get("launch_status", "planned"),
            "llm_provider": "none",
            "lora_training_enabled": False,
            "matrix": request.matrix,
            "methods": list(request.methods),
            "output_dir": str(output_dir),
            "protocol_version": "protocol_v1",
            "python_executable": sys.executable,
            "seed": request.seed,
        },
    )


def _prepare_method_dir(
    method_dir: Path,
    bundle: DatasetBundle,
    method: str,
    request: PaperMatrixRequest,
    experiment: dict[str, Any],
) -> None:
    ensure_dir(method_dir)
    ensure_dir(method_dir / "checkpoints")
    for stale_name in (
        "failure_report.json",
        "metrics.csv",
        "metrics.json",
        "predictions.jsonl",
        "runtime.json",
        "training_metrics.json",
    ):
        stale = method_dir / stale_name
        if stale.is_file():
            stale.unlink()
    resolved = {
        "api_calls_allowed": False,
        "candidate_artifact": str(bundle.candidate_artifact),
        "candidate_protocol": bundle.candidate_protocol,
        "dataset": bundle.name,
        "dataset_config": str(bundle.config_path),
        "experiment": experiment,
        "history_source": "train_split_only",
        "llm": {"allow_api_calls": False, "provider": "none"},
        "lora_training_enabled": False,
        "matrix": request.matrix,
        "method": method,
        "output_dir": str(method_dir),
        "protocol_version": "protocol_v1",
        "seed": request.seed,
        "split_artifact": str(bundle.split_artifact),
    }
    save_resolved_config(resolved, method_dir / "resolved_config.yaml")
    write_json(method_dir / "environment.json", collect_environment(resolve_path(".")))
    write_json(method_dir / "artifact_checksums.json", bundle.artifact_checksums)
    _append_log(method_dir, f"starting method={method} dataset={bundle.name}")


def _resume_completed_method(
    method_dir: Path,
    dataset: str,
    method: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    metrics_path = method_dir / "metrics.json"
    predictions_path = method_dir / "predictions.jsonl"
    if not metrics_path.is_file() or not predictions_path.is_file():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    runtime_path = method_dir / "runtime.json"
    runtime = {}
    if runtime_path.is_file():
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    checkpoint_paths = sorted((method_dir / "checkpoints").glob("*")) if (method_dir / "checkpoints").is_dir() else []
    status_row = {
        "checkpoint_path": str(checkpoint_paths[0]) if checkpoint_paths else "",
        "dataset": dataset,
        "failure_reason": "",
        "message": "resumed existing succeeded artifacts",
        "method": method,
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "runtime_seconds": runtime.get("runtime_seconds", metrics.get("overall", {}).get("runtime_seconds", "")),
        "status": "succeeded",
    }
    _append_log(method_dir, "resumed existing succeeded artifacts")
    return status_row, metrics


def _append_log(method_dir: Path, message: str) -> None:
    with (method_dir / "logs.txt").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}\n")


def _metrics_row(dataset: str, method: str, metrics: dict[str, Any]) -> dict[str, Any]:
    overall = dict(metrics.get("overall", {}))
    row: dict[str, Any] = {
        "dataset": dataset,
        "method": method,
        "num_predictions": metrics.get("num_predictions", 0),
        "seed": metrics.get("seed", 0),
    }
    for key, value in sorted(overall.items()):
        if isinstance(value, (int, float)):
            row[key] = value
    return row


def _dataset_metric_rows(method_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in method_rows:
        by_dataset[str(row.get("dataset", ""))].append(row)
    output: list[dict[str, Any]] = []
    for dataset, rows in sorted(by_dataset.items()):
        merged: dict[str, Any] = {
            "dataset": dataset,
            "method_count": len(rows),
            "seed": rows[0].get("seed", 0) if rows else 0,
        }
        numeric_keys = sorted(
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and key not in {"seed"}
        )
        for key in numeric_keys:
            values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
            if values:
                merged[f"mean_{key}"] = sum(values) / float(len(values))
        output.append(merged)
    return output


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _latex_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "\\begin{tabular}{ll}\ndataset & note \\\\\n\\end{tabular}\n"
    lines = ["\\begin{tabular}{" + "l" * len(columns) + "}", " & ".join(columns) + " \\\\"]
    for row in rows:
        lines.append(" & ".join(_format_latex_value(row.get(column, "")) for column in columns) + " \\\\")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def _format_latex_value(value: Any) -> str:
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return value.replace("_", "\\_")
        return f"{number:.6f}"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("_", "\\_")


def _write_method_artifact(method_dir: Path, name: str, payload: Any) -> None:
    write_json(method_dir / name, payload)


def _item_text(row: dict[str, Any]) -> str:
    values = [
        row.get("title"),
        row.get("description"),
        row.get("category"),
        row.get("brand"),
        row.get("raw_text"),
    ]
    return " ".join(str(value) for value in values if value not in (None, ""))


def _train_sequences(train_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        by_user[str(row["user_id"])].append(row)
    return {
        user: [
            str(row["item_id"])
            for row in sorted(
                rows,
                key=lambda value: (
                    float(value["timestamp"]) if value.get("timestamp") is not None else -1.0,
                    str(value["item_id"]),
                ),
            )
        ]
        for user, rows in by_user.items()
    }


def _sasrec_training_examples(
    sequences: dict[str, list[str]],
    item_to_idx: dict[str, int],
    max_seq_len: int,
) -> list[tuple[list[int], int]]:
    examples: list[tuple[list[int], int]] = []
    for items in sequences.values():
        indices = [item_to_idx[item] for item in items if item in item_to_idx]
        for position in range(1, len(indices)):
            examples.append((left_pad(indices[:position], max_seq_len), indices[position]))
    return examples


def _batched(values: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), int(batch_size)):
        yield values[start : start + int(batch_size)]


def _require_torch(label: str) -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - covered by environment checks.
        raise RuntimeError(f"PyTorch unavailable for {label}") from exc
    return torch


def _resource_guard(bundle: DatasetBundle, method: str) -> None:
    if len(bundle.test_timestamp_by_user) > 100000:
        raise RuntimeError(
            "Local Phase 9B resource guard: this dataset has "
            f"{len(bundle.test_timestamp_by_user)} test users and the current prediction schema "
            "requires expanding the frozen 1000-candidate shared pool into every prediction row. "
            f"{method} on {bundle.name} is recorded as an explicit resource failure; "
            "rerun the failed Amazon jobs after approving a compact shared-pool prediction schema "
            "or using a higher-throughput experiment machine."
        )
    trainable = {"mf_bpr", "sasrec", "temporal_graph_encoder", "time_graph_evidence_dynamic"}
    if method not in trainable:
        return
    if len(bundle.test_timestamp_by_user) <= 100000:
        return
    torch = _require_torch(method)
    if bool(getattr(torch.cuda, "is_available", lambda: False)()):
        return
    raise RuntimeError(
        "PyTorch is available, but only CPU was detected; "
        f"{method} on {bundle.name} has {len(bundle.test_timestamp_by_user)} test users and is "
        "recorded as an explicit Phase 9B resource failure instead of silently skipping. "
        "Rerun failed jobs with a GPU-enabled PyTorch interpreter."
    )
