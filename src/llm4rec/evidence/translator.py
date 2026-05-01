"""Graph-to-text evidence translation without free-form hallucination."""

from __future__ import annotations

import json
from typing import Iterable

from llm4rec.evidence.base import Evidence


TRANSLATOR_MODES = {"compact", "verbose", "prompt_ready_json", "contrastive"}


class GraphToTextTranslator:
    """Translate evidence objects using only fields already present in evidence."""

    def __init__(self, mode: str = "compact") -> None:
        if mode not in TRANSLATOR_MODES:
            raise ValueError(f"Unsupported translator mode: {mode}")
        self.mode = mode

    def translate(self, evidence: Evidence | Iterable[Evidence]) -> str:
        """Translate a single evidence object or an evidence list."""

        if isinstance(evidence, Evidence):
            rows = [evidence]
        else:
            rows = list(evidence)
        if self.mode == "prompt_ready_json":
            return json.dumps(
                {"evidence": [row.to_dict() for row in rows]},
                ensure_ascii=True,
                sort_keys=True,
            )
        parts = [self._translate_one(row) for row in rows]
        return "\n".join(part for part in parts if part)

    def _translate_one(self, evidence: Evidence) -> str:
        source = _item_label(evidence, "source")
        target = _item_label(evidence, "target")
        stats = evidence.stats
        gap_bucket = evidence.timestamp_info.get("gap_bucket")
        mean_gap = evidence.timestamp_info.get("mean_gap_seconds")
        count = stats.get("transition_count")
        window_score = stats.get("time_window_score")
        similarity = stats.get("semantic_similarity")

        if self.mode == "contrastive" or evidence.evidence_type == "contrastive":
            fragments = [f"{target} is contrasted with {source}"]
            if similarity is not None:
                fragments.append(f"semantic_similarity={_fmt(similarity)}")
            if count:
                fragments.append(f"transition_count={int(count)}")
            if gap_bucket:
                fragments.append(f"gap_bucket={gap_bucket}")
            return "; ".join(fragments) + "."

        if evidence.evidence_type == "transition":
            fragments = [f"After {source}, users reached {target}"]
            if count is not None:
                fragments.append(f"transition_count={int(count)}")
            if mean_gap is not None:
                fragments.append(f"mean_gap_seconds={_fmt(mean_gap)}")
            if gap_bucket:
                fragments.append(f"gap_bucket={gap_bucket}")
            return _sentence(fragments)

        if evidence.evidence_type == "time_window":
            fragments = [f"{source} and {target} appeared in the configured time window"]
            if window_score is not None:
                fragments.append(f"time_window_score={_fmt(window_score)}")
            if gap_bucket:
                fragments.append(f"gap_bucket={gap_bucket}")
            return _sentence(fragments)

        if evidence.evidence_type == "semantic":
            fragments = [f"{source} and {target} share item metadata evidence"]
            if similarity is not None:
                fragments.append(f"semantic_similarity={_fmt(similarity)}")
            return _sentence(fragments)

        if evidence.evidence_type == "history":
            recency = stats.get("recent_signal")
            fragments = [f"{source} is in the user's history before considering {target}"]
            if recency is not None:
                fragments.append(f"recent_signal={_fmt(recency)}")
            return _sentence(fragments)

        if evidence.evidence_type == "user_drift":
            fragments = [f"History category drift supports checking {target}"]
            drift_from = evidence.metadata.get("drift_from")
            drift_to = evidence.metadata.get("drift_to")
            if drift_from is not None:
                fragments.append(f"drift_from={drift_from}")
            if drift_to is not None:
                fragments.append(f"drift_to={drift_to}")
            return _sentence(fragments)

        return evidence.text


def _item_label(evidence: Evidence, side: str) -> str:
    item_id = evidence.source_item if side == "source" else evidence.target_item
    title = evidence.metadata.get(f"{side}_title")
    if title:
        return f"{title} ({item_id})"
    return str(item_id)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _sentence(fragments: list[str]) -> str:
    return "; ".join(fragments) + "."
