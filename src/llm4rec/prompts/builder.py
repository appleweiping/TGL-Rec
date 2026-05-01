"""Prompt builder for sequence/time LLM diagnostic variants."""

from __future__ import annotations

from typing import Any

from llm4rec.data.time_features import consecutive_time_gaps
from llm4rec.prompts.base import PromptBuildError, PromptExample, PromptRequest
from llm4rec.prompts.templates import (
    NO_HALLUCINATION_INSTRUCTION,
    STRICT_JSON_INSTRUCTION,
    TASK_HEADER,
)
from llm4rec.prompts.variants import get_prompt_variant


def build_prompt(
    example: PromptExample,
    *,
    prompt_variant: str,
    max_history_items: int = 20,
    max_evidence_items: int = 8,
) -> PromptRequest:
    """Build a deterministic prompt for one diagnostic variant."""

    spec = get_prompt_variant(prompt_variant)
    if not example.candidate_items:
        raise PromptBuildError("candidate_items must not be empty")
    candidate_ids = [str(item) for item in example.candidate_items]
    missing = [item for item in candidate_ids if item not in example.item_records]
    if missing:
        raise PromptBuildError(f"Missing item records for candidates: {missing[:5]}")

    history = [str(item) for item in example.history][-max_history_items:]
    lines = [
        TASK_HEADER,
        f"prompt_version: {spec.prompt_version}",
        f"user_id: {example.user_id}",
        NO_HALLUCINATION_INSTRUCTION,
        "",
        "Candidate items:",
        *_format_candidates(candidate_ids, example.item_records),
        "",
        "User history:",
    ]
    lines.extend(
        _format_history(
            history,
            example=example,
            uses_order=spec.uses_order,
            uses_time_gaps=spec.uses_time_gaps,
            uses_time_buckets=spec.uses_time_buckets,
        )
    )
    if spec.uses_transition_evidence:
        lines.extend(["", "Directed transition evidence:"])
        lines.extend(_format_transition_evidence(example.transition_evidence[:max_evidence_items]))
    if spec.uses_time_window_evidence:
        lines.extend(["", "Time-window evidence:"])
        lines.extend(_format_time_window_evidence(example.time_window_evidence[:max_evidence_items]))
    if spec.uses_contrastive_evidence:
        lines.extend(["", "Semantic-vs-transition contrast evidence:"])
        lines.extend(_format_contrastive_evidence(example.contrastive_evidence[:max_evidence_items]))
    lines.extend(["", STRICT_JSON_INSTRUCTION])
    prompt = "\n".join(lines).strip() + "\n"
    return PromptRequest(
        prompt=prompt,
        prompt_variant=spec.name,
        prompt_version=spec.prompt_version,
        candidate_item_ids=candidate_ids,
        metadata={
            "target_item": example.target_item,
            "user_id": example.user_id,
            "variant_uses_order": spec.uses_order,
            "variant_uses_time_gaps": spec.uses_time_gaps,
            "variant_uses_time_buckets": spec.uses_time_buckets,
            "variant_uses_transition_evidence": spec.uses_transition_evidence,
            "variant_uses_time_window_evidence": spec.uses_time_window_evidence,
            "variant_uses_contrastive_evidence": spec.uses_contrastive_evidence,
        },
    )


def _format_candidates(
    candidate_ids: list[str],
    item_records: dict[str, dict[str, Any]],
) -> list[str]:
    return [
        f"- {item_id}: {_title(item_records[item_id])}"
        for item_id in candidate_ids
    ]


def _format_history(
    history: list[str],
    *,
    example: PromptExample,
    uses_order: bool,
    uses_time_gaps: bool,
    uses_time_buckets: bool,
) -> list[str]:
    if not history:
        return ["- <empty history>"]
    item_records = example.item_records
    if not uses_order and not uses_time_gaps and not uses_time_buckets:
        titles = [f"{item_id}: {_title(item_records.get(item_id, {}))}" for item_id in history]
        return ["- " + "; ".join(titles)]
    if uses_time_gaps:
        return [_format_gap_chain(example)]
    if uses_time_buckets:
        return [_format_bucket_chain(example)]
    return [
        f"{index}. {item_id}: {_title(item_records.get(item_id, {}))}"
        for index, item_id in enumerate(history, start=1)
    ]


def _format_gap_chain(example: PromptExample) -> str:
    rows = _time_rows(example)
    parts: list[str] = []
    for row in rows:
        item_id = str(row["item_id"])
        title = _title(example.item_records.get(item_id, {}))
        gap_seconds = row.get("gap_seconds")
        if not parts:
            parts.append(f"{item_id}: {title}")
        else:
            parts.append(f"after {_human_gap(gap_seconds)} -> {item_id}: {title}")
    return "- " + " -> ".join(parts)


def _format_bucket_chain(example: PromptExample) -> str:
    rows = _time_rows(example)
    parts: list[str] = []
    for row in rows:
        item_id = str(row["item_id"])
        title = _title(example.item_records.get(item_id, {}))
        bucket = str(row.get("gap_bucket", "unknown"))
        if not parts:
            parts.append(f"{item_id}: {title}")
        else:
            parts.append(f"[{bucket}] {item_id}: {title}")
    return "- " + " -> ".join(parts)


def _format_transition_evidence(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- none"]
    return [
        "- Users who watched {source} often watched {target} next; "
        "transition_count={count}; median_gap={median}; dominant_bucket={bucket}.".format(
            source=row.get("source_item"),
            target=row.get("target_item"),
            count=row.get("transition_count", row.get("count", 0)),
            median=row.get("median_time_gap"),
            bucket=row.get("dominant_gap_bucket", _dominant_bucket(row.get("bucket_counts", {}))),
        )
        for row in rows
    ]


def _format_time_window_evidence(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- none"]
    output = []
    for row in rows:
        source = row.get("source_item")
        target = row.get("target_item")
        parts = []
        for key in ("time_window_score_1d", "time_window_score_7d", "time_window_score_30d"):
            if row.get(key) not in (None, ""):
                parts.append(f"{key.removeprefix('time_window_score_')}={row[key]}")
        if row.get("window"):
            parts.append(f"window={row['window']}")
        output.append(f"- {source} and {target} frequently appear within windows; {'; '.join(parts)}.")
    return output


def _format_contrastive_evidence(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- none"]
    return [
        "- {target} has text_similarity={similarity} to {source}, transition_score={transition}, "
        "same_category={same_category}, primary_group={group}.".format(
            source=row.get("source_item"),
            target=row.get("target_item"),
            similarity=row.get("text_similarity"),
            transition=row.get("transition_score", row.get("transition_count", 0)),
            same_category=row.get("same_genre_or_category"),
            group=row.get("primary_group", row.get("group", "unknown")),
        )
        for row in rows
    ]


def _time_rows(example: PromptExample) -> list[dict[str, Any]]:
    rows = [row for row in example.history_rows if str(row.get("item_id")) in set(example.history)]
    if rows:
        return consecutive_time_gaps(rows)[-len(example.history) :]
    return [
        {
            "gap_bucket": "unknown",
            "gap_seconds": None,
            "item_id": item_id,
            "timestamp": None,
            "user_id": example.user_id,
        }
        for item_id in example.history
    ]


def _title(row: dict[str, Any]) -> str:
    title = row.get("title") or row.get("raw_text") or "<unknown title>"
    return str(title).replace("\n", " ")


def _human_gap(value: Any) -> str:
    if value is None:
        return "unknown gap"
    seconds = float(value)
    if seconds < 3600:
        return f"{seconds:.0f} seconds"
    if seconds < 86400:
        return f"{seconds / 3600.0:.1f} hours"
    return f"{seconds / 86400.0:.1f} days"


def _dominant_bucket(bucket_counts: dict[str, Any]) -> str:
    if not bucket_counts:
        return "unknown"
    return sorted(bucket_counts, key=lambda key: (-int(bucket_counts[key]), str(key)))[0]

