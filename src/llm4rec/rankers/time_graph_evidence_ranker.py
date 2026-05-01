"""Smoke ranker for TimeGraphEvidenceRec skeleton."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from llm4rec.evidence.retriever import TemporalEvidenceRetriever
from llm4rec.evidence.scorer import score_candidates
from llm4rec.evidence.temporal_graph import build_temporal_graph_artifacts
from llm4rec.evidence.translator import GraphToTextTranslator
from llm4rec.experiments.config import resolve_path
from llm4rec.io.artifacts import write_json, write_jsonl
from llm4rec.methods.ablation import AblationSwitches
from llm4rec.methods.config import load_method_config
from llm4rec.methods.leakage import LeakageValidator, validate_evidence_list
from llm4rec.rankers.base import RankingExample, RankingResult, result_from_scores


class TimeGraphEvidenceRanker:
    """Deterministic Phase 5 skeleton ranker backed by temporal graph evidence."""

    name = "time_graph_evidence_rec"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        candidate_protocol: str = "full_catalog",
    ) -> None:
        self.config = load_method_config(config or {})
        self.method = dict(self.config.get("method", {}))
        self.name = str(self.method.get("name", self.name))
        self.reportable = bool(self.method.get("reportable", False))
        self.candidate_protocol = candidate_protocol
        self.ablation = AblationSwitches.from_dict(dict(self.config.get("ablation", {})))
        self.translator = GraphToTextTranslator(str(self.config.get("translator", {}).get("mode", "prompt_ready_json")))
        self.validator = LeakageValidator(reportable=self.reportable)
        self.dynamic_encoder = None
        self.dynamic_encoder_status = self._load_dynamic_encoder()
        self.retriever: TemporalEvidenceRetriever | None = None
        self.transition_edges: list[dict[str, Any]] = []
        self.time_window_edges: list[dict[str, Any]] = []
        self.item_records: list[dict[str, Any]] = []

    def fit(
        self,
        train_interactions: list[dict[str, Any]],
        item_records: list[dict[str, Any]],
    ) -> None:
        """Fit by building train-only temporal graph artifacts in memory."""

        self.validator.validate_config(self.config)
        self.item_records = [dict(row) for row in item_records]
        window_seconds = float(self.config.get("time_window", {}).get("window_seconds", 86400))
        graph = build_temporal_graph_artifacts(
            train_interactions=train_interactions,
            output_dir=Path("outputs") / "cache" / "phase5_in_memory_graph",
            window_seconds=window_seconds,
            candidate_protocol=self.candidate_protocol,
        )
        self.transition_edges = list(graph["transition_edges"])
        self.time_window_edges = list(graph["time_window_edges"])
        self.retriever = TemporalEvidenceRetriever(
            transition_edges=self.transition_edges if self.ablation.use_transition_edges else [],
            time_window_edges=self.time_window_edges if self.ablation.use_time_window_edges else [],
            item_records=self.item_records,
            config={**dict(self.config.get("retrieval", {})), "modes": self._enabled_modes()},
            transition_artifact=str(graph["transition_path"]),
            time_window_artifact=str(graph["time_window_path"]),
            candidate_protocol=self.candidate_protocol,
            constructed_from="train_only",
        )

    def rank(self, example: RankingExample) -> RankingResult:
        """Rank candidates and attach evidence in metadata."""

        self.validator.validate_example(
            history=example.history,
            target_item=example.target_item,
            candidate_items=example.candidate_items,
            prediction_timestamp=example.metadata.get("prediction_timestamp"),
            candidate_metadata=example.metadata,
        )
        evidence = []
        retrieval_metadata: dict[str, Any] = {}
        warnings: list[str] = []
        if self.ablation.use_retrieval and self.retriever is not None:
            retrieval = self.retriever.retrieve(
                user_id=example.user_id,
                history=example.history,
                candidate_items=example.candidate_items,
                prediction_timestamp=example.metadata.get("prediction_timestamp"),
            )
            evidence = retrieval.evidence
            retrieval_metadata = retrieval.metadata
            warnings = retrieval.warnings
            if not self.ablation.use_time_gap_tags:
                evidence = [_strip_gap_tags(row) for row in evidence]
            validate_evidence_list(
                evidence,
                reportable=self.reportable,
                prediction_timestamp=example.metadata.get("prediction_timestamp"),
            )

        scores_by_item = score_candidates(
            evidence,
            example.candidate_items,
            dict(self.config.get("scoring", {})),
        )
        dynamic_scores: dict[str, float] = {}
        if self.ablation.use_dynamic_encoder and self.dynamic_encoder is not None:
            weight = float(self.config.get("scoring", {}).get("dynamic_encoder_weight", 0.0))
            for item_id in example.candidate_items:
                score = float(
                    self.dynamic_encoder.score(
                        example.user_id,
                        str(item_id),
                        example.metadata.get("prediction_timestamp"),
                    )
                )
                dynamic_scores[str(item_id)] = score
                scores_by_item[str(item_id)] = float(scores_by_item.get(str(item_id), 0.0)) + weight * score
        result = result_from_scores(
            example=example,
            scores_by_item=scores_by_item,
            metadata={
                "ablation": self.ablation.to_dict(),
                "dynamic_encoder_enabled": bool(self.ablation.use_dynamic_encoder),
                "dynamic_encoder_scores": dynamic_scores,
                "dynamic_encoder_status": self.dynamic_encoder_status,
                "evidence_count": len(evidence),
                "evidence_used": [row.to_dict() for row in evidence],
                "non_reportable_phase5": True,
                "prompt_ready_evidence": self.translator.translate(evidence)
                if self.ablation.use_explanation
                else "",
                "retrieval": retrieval_metadata,
                "retrieval_warnings": warnings,
            },
        )
        return result

    def save_artifact(self, output_dir: str | Path) -> None:
        """Persist graph artifacts and method config."""

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        write_jsonl(output / "transition_edges.jsonl", self.transition_edges)
        write_jsonl(output / "time_window_edges.jsonl", self.time_window_edges)
        write_json(output / "method_config.json", self.config)

    def _enabled_modes(self) -> list[str]:
        modes = [str(mode) for mode in self.config.get("retrieval", {}).get("modes", [])]
        if not self.ablation.use_retrieval:
            return []
        disabled: set[str] = set()
        if not self.ablation.use_temporal_graph:
            disabled.update({"transition_topk", "time_window_topk", "contrastive_transition_only"})
        if not self.ablation.use_transition_edges:
            disabled.update({"transition_topk", "contrastive_transition_only"})
        if not self.ablation.use_time_window_edges:
            disabled.add("time_window_topk")
        if not self.ablation.use_semantic_similarity:
            disabled.update({"semantic_topk", "contrastive_transition_only"})
        if not self.ablation.use_recent_block:
            disabled.add("recent_history_focused")
        if not self.ablation.use_long_term_block:
            disabled.add("user_drift_blocks")
        return [mode for mode in modes if mode not in disabled]

    def _load_dynamic_encoder(self) -> dict[str, Any]:
        encoder_config = dict(self.config.get("encoder", {}))
        if not self.ablation.use_dynamic_encoder:
            return {"enabled": False, "reason": "ablation_disabled"}
        if str(encoder_config.get("type", "")) != "temporal_graph_encoder":
            return {"enabled": False, "reason": "unsupported_encoder_type"}
        checkpoint = encoder_config.get("checkpoint_path")
        if not checkpoint:
            return {"enabled": False, "reason": "missing_checkpoint_path"}
        try:
            from llm4rec.encoders.temporal_graph_encoder import TORCH_AVAILABLE, TemporalGraphEncoder

            if not TORCH_AVAILABLE:
                return {"enabled": False, "reason": "pytorch_unavailable"}
            checkpoint_path = resolve_path(str(checkpoint))
            if not checkpoint_path.is_file():
                return {"enabled": False, "reason": "missing_checkpoint", "checkpoint_path": str(checkpoint_path)}
            self.dynamic_encoder = TemporalGraphEncoder.load(checkpoint_path)
            return {"enabled": True, "checkpoint_path": str(checkpoint_path), "type": "temporal_graph_encoder"}
        except Exception as exc:
            return {"enabled": False, "reason": type(exc).__name__, "message": str(exc)}


def prediction_row_from_result(
    *,
    example: RankingExample,
    result: RankingResult,
    method_name: str,
    phase: str = "phase5_method_smoke",
) -> dict[str, Any]:
    """Convert a ranking result to the shared prediction JSONL schema."""

    return {
        "candidate_items": example.candidate_items,
        "domain": example.domain,
        "metadata": {
            **result.metadata,
            "eval_split": example.metadata.get("split"),
            "phase": phase,
            "reportable": False,
        },
        "method": method_name,
        "predicted_items": result.items,
        "raw_output": result.raw_output,
        "scores": result.scores,
        "target_item": example.target_item,
        "user_id": example.user_id,
    }


def _strip_gap_tags(row):
    timestamp_info = dict(row.timestamp_info)
    timestamp_info["gap_bucket"] = None
    return replace(row, timestamp_info=timestamp_info)
