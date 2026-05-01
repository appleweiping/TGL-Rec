"""Shared prompt text blocks for Phase 3A diagnostics."""

from __future__ import annotations

STRICT_JSON_INSTRUCTION = """Return strict JSON only with this schema:
{
  "ranked_item_ids": ["candidate_id_1", "candidate_id_2"],
  "reasoning_summary": "one short grounded explanation",
  "evidence_used": [
    {
      "type": "history|time_gap|time_bucket|transition|time_window|semantic",
      "source_item": "candidate_or_history_item_id",
      "target_item": "candidate_item_id",
      "text": "short evidence statement"
    }
  ]
}
Do not include markdown. Do not recommend any item ID outside the candidate set."""

NO_HALLUCINATION_INSTRUCTION = (
    "No-hallucination rule: every ranked_item_ids entry must be one of the candidate item IDs "
    "listed below. If uncertain, still rank only candidate IDs."
)

TASK_HEADER = (
    "You are running a diagnostic reranking task for LLM4Rec sequence/time sensitivity. "
    "Rank the candidate items for the user's next interaction."
)

