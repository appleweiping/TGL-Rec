"""Temporal evidence retrieval over train-only graph artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm4rec.evidence.base import Evidence
from llm4rec.evidence.temporal_graph import (
    make_contrastive_evidence,
    make_history_evidence,
    make_semantic_evidence,
    make_user_drift_evidence,
    time_window_edge_to_evidence,
    transition_edge_to_evidence,
)


RETRIEVAL_MODES = {
    "transition_topk",
    "time_window_topk",
    "semantic_topk",
    "contrastive_transition_only",
    "recent_history_focused",
    "user_drift_blocks",
}


@dataclass(frozen=True)
class RetrievalResult:
    """Evidence retrieval output."""

    evidence: list[Evidence]
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class TemporalEvidenceRetriever:
    """Retrieve candidate-grounded evidence without using target labels."""

    def __init__(
        self,
        *,
        transition_edges: list[dict[str, Any]] | None = None,
        time_window_edges: list[dict[str, Any]] | None = None,
        item_records: list[dict[str, Any]] | None = None,
        config: dict[str, Any] | None = None,
        transition_artifact: str = "transition_edges.jsonl",
        time_window_artifact: str = "time_window_edges.jsonl",
        candidate_protocol: str = "full_catalog",
        constructed_from: str = "train_only",
    ) -> None:
        self.config = dict(config or {})
        modes = self.config.get("modes", ["transition_topk", "time_window_topk"])
        self.modes = [str(mode) for mode in modes]
        unknown = sorted(set(self.modes) - RETRIEVAL_MODES)
        if unknown:
            raise ValueError(f"Unsupported retrieval modes: {unknown}")
        self.top_k_per_candidate = int(self.config.get("top_k_per_candidate", 3))
        self.recent_history_items = int(self.config.get("recent_history_items", 2))
        self.semantic_min_similarity = float(self.config.get("semantic_min_similarity", 0.1))
        self.contrastive_max_similarity = float(self.config.get("contrastive_max_similarity", 0.35))
        self.transition_artifact = transition_artifact
        self.time_window_artifact = time_window_artifact
        self.candidate_protocol = candidate_protocol
        self.constructed_from = constructed_from
        self.item_records = {str(row["item_id"]): dict(row) for row in item_records or []}
        self.transition_index = _edge_index(transition_edges or [], directed=True)
        self.time_window_index = _edge_index(time_window_edges or [], directed=True)

    def retrieve(
        self,
        *,
        user_id: str,
        history: list[str],
        candidate_items: list[str],
        prediction_timestamp: float | None = None,
    ) -> RetrievalResult:
        """Retrieve evidence for a user history and candidate set."""

        del prediction_timestamp  # Reserved for stricter temporal filtering hooks.
        evidence: list[Evidence] = []
        warnings: list[str] = []
        history_items = [str(item) for item in history]
        candidates = [str(item) for item in candidate_items]
        if not history_items:
            warnings.append("empty_history")
        if not candidates:
            warnings.append("empty_candidate_set")
        recent_sources = list(reversed(history_items[-self.recent_history_items :]))

        for candidate in candidates:
            candidate_evidence: list[Evidence] = []
            if "transition_topk" in self.modes:
                candidate_evidence.extend(self._transition_evidence(recent_sources, candidate))
            if "time_window_topk" in self.modes:
                candidate_evidence.extend(self._time_window_evidence(recent_sources, candidate))
            if "semantic_topk" in self.modes:
                candidate_evidence.extend(self._semantic_evidence(recent_sources, candidate))
            if "contrastive_transition_only" in self.modes:
                candidate_evidence.extend(self._contrastive_evidence(recent_sources, candidate))
            if "recent_history_focused" in self.modes:
                candidate_evidence.extend(self._history_evidence(recent_sources, candidate))
            if "user_drift_blocks" in self.modes:
                candidate_evidence.extend(self._user_drift_evidence(history_items, candidate))
            if not candidate_evidence:
                warnings.append(f"missing_evidence:{candidate}")
            evidence.extend(_dedupe_evidence(candidate_evidence)[: self.top_k_per_candidate])

        return RetrievalResult(
            evidence=_dedupe_evidence(evidence),
            metadata={
                "candidate_count": len(candidates),
                "constructed_from": self.constructed_from,
                "history_length": len(history_items),
                "modes": self.modes,
                "top_k_per_candidate": self.top_k_per_candidate,
                "user_id": str(user_id),
            },
            warnings=sorted(set(warnings)),
        )

    def _transition_evidence(self, sources: list[str], candidate: str) -> list[Evidence]:
        rows: list[Evidence] = []
        for source in sources:
            edge = self.transition_index.get((source, candidate))
            if edge is None:
                continue
            rows.append(
                transition_edge_to_evidence(
                    edge,
                    graph_artifact=self.transition_artifact,
                    candidate_protocol=self.candidate_protocol,
                    constructed_from=self.constructed_from,
                    metadata=_item_metadata(self.item_records, source, candidate),
                )
            )
        return sorted(rows, key=lambda row: (-float(row.stats.get("transition_count") or 0.0), row.evidence_id))

    def _time_window_evidence(self, sources: list[str], candidate: str) -> list[Evidence]:
        rows: list[Evidence] = []
        for source in sources:
            edge = self.time_window_index.get((source, candidate))
            if edge is None:
                continue
            rows.append(
                time_window_edge_to_evidence(
                    edge,
                    graph_artifact=self.time_window_artifact,
                    candidate_protocol=self.candidate_protocol,
                    constructed_from=self.constructed_from,
                    metadata=_item_metadata(self.item_records, source, candidate),
                )
            )
        return sorted(rows, key=lambda row: (-float(row.stats.get("time_window_score") or 0.0), row.evidence_id))

    def _semantic_evidence(self, sources: list[str], candidate: str) -> list[Evidence]:
        rows: list[Evidence] = []
        for source in sources:
            similarity = _semantic_similarity(self.item_records.get(source), self.item_records.get(candidate))
            if similarity < self.semantic_min_similarity:
                continue
            rows.append(
                make_semantic_evidence(
                    source_item=source,
                    target_item=candidate,
                    similarity=similarity,
                    candidate_protocol=self.candidate_protocol,
                    constructed_from=self.constructed_from,
                    metadata=_item_metadata(self.item_records, source, candidate),
                )
            )
        return sorted(rows, key=lambda row: (-float(row.stats.get("semantic_similarity") or 0.0), row.evidence_id))

    def _contrastive_evidence(self, sources: list[str], candidate: str) -> list[Evidence]:
        rows: list[Evidence] = []
        for source in sources:
            edge = self.transition_index.get((source, candidate))
            if edge is None:
                continue
            similarity = _semantic_similarity(self.item_records.get(source), self.item_records.get(candidate))
            if similarity > self.contrastive_max_similarity:
                continue
            rows.append(
                make_contrastive_evidence(
                    source_item=source,
                    target_item=candidate,
                    transition_count=int(edge.get("count", 0)),
                    semantic_similarity=similarity,
                    candidate_protocol=self.candidate_protocol,
                    constructed_from=self.constructed_from,
                    metadata=_item_metadata(self.item_records, source, candidate),
                )
            )
        return sorted(rows, key=lambda row: (-float(row.stats.get("transition_count") or 0.0), row.evidence_id))

    def _history_evidence(self, sources: list[str], candidate: str) -> list[Evidence]:
        rows: list[Evidence] = []
        for rank, source in enumerate(sources, start=1):
            rows.append(
                make_history_evidence(
                    source_item=source,
                    target_item=candidate,
                    recent_rank=rank,
                    candidate_protocol=self.candidate_protocol,
                    constructed_from=self.constructed_from,
                    metadata=_item_metadata(self.item_records, source, candidate),
                )
            )
        return rows

    def _user_drift_evidence(self, history: list[str], candidate: str) -> list[Evidence]:
        if len(history) < 2:
            return []
        midpoint = len(history) // 2
        long_term = _dominant_category(self.item_records, history[:midpoint])
        recent = _dominant_category(self.item_records, history[midpoint:])
        candidate_category = self.item_records.get(candidate, {}).get("category")
        if long_term == recent or candidate_category != recent:
            return []
        source = history[-1]
        return [
            make_user_drift_evidence(
                source_item=source,
                target_item=candidate,
                candidate_protocol=self.candidate_protocol,
                drift_from=long_term,
                drift_to=recent,
                constructed_from=self.constructed_from,
                metadata=_item_metadata(self.item_records, source, candidate),
            )
        ]


def _edge_index(edges: list[dict[str, Any]], *, directed: bool) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in edges:
        source = str(edge["source_item"])
        target = str(edge["target_item"])
        index[(source, target)] = edge
        if not directed or not bool(edge.get("directed", True)):
            index[(target, source)] = edge
    return index


def _item_metadata(
    item_records: dict[str, dict[str, Any]],
    source: str,
    target: str,
) -> dict[str, Any]:
    source_row = item_records.get(source, {})
    target_row = item_records.get(target, {})
    return {
        "source_category": source_row.get("category"),
        "source_title": source_row.get("title"),
        "target_category": target_row.get("category"),
        "target_title": target_row.get("title"),
    }


def _semantic_similarity(left: dict[str, Any] | None, right: dict[str, Any] | None) -> float:
    if not left or not right:
        return 0.0
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / float(len(left_tokens | right_tokens))


def _tokens(row: dict[str, Any]) -> set[str]:
    values = [row.get("title"), row.get("description"), row.get("category"), row.get("raw_text")]
    tokens: set[str] = set()
    for value in values:
        if value is None:
            continue
        for token in str(value).lower().replace("_", " ").split():
            cleaned = "".join(ch for ch in token if ch.isalnum())
            if cleaned:
                tokens.add(cleaned)
    return tokens


def _dominant_category(item_records: dict[str, dict[str, Any]], items: list[str]) -> str | None:
    counts: dict[str, int] = {}
    for item in items:
        category = item_records.get(item, {}).get("category")
        if category is None:
            continue
        counts[str(category)] = counts.get(str(category), 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _dedupe_evidence(evidence: list[Evidence]) -> list[Evidence]:
    seen: set[str] = set()
    output: list[Evidence] = []
    for row in evidence:
        if row.evidence_id in seen:
            continue
        seen.add(row.evidence_id)
        output.append(row)
    return output
