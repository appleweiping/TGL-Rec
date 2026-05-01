from llm4rec.prompts.base import PromptExample
from llm4rec.prompts.builder import build_prompt


def _example() -> PromptExample:
    records = {
        "i1": {"item_id": "i1", "title": "Alpha", "category": "A"},
        "i2": {"item_id": "i2", "title": "Beta", "category": "B"},
        "i3": {"item_id": "i3", "title": "Gamma", "category": "C"},
    }
    return PromptExample(
        user_id="u1",
        history=["i1", "i2"],
        target_item="i3",
        candidate_items=["i2", "i3"],
        item_records=records,
        history_rows=[
            {"user_id": "u1", "item_id": "i1", "timestamp": 0},
            {"user_id": "u1", "item_id": "i2", "timestamp": 7200},
        ],
        transition_evidence=[
            {
                "source_item": "i2",
                "target_item": "i3",
                "transition_count": 3,
                "median_time_gap": 3600,
                "dominant_gap_bucket": "same_session",
            }
        ],
    )


def test_prompt_builder_includes_candidates_and_strict_json_instruction():
    request = build_prompt(_example(), prompt_variant="history_with_transition_evidence")

    assert "prompt_version: phase3a.history_with_transition_evidence.v1" in request.prompt
    assert "i2: Beta" in request.prompt
    assert "i3: Gamma" in request.prompt
    assert "No-hallucination rule" in request.prompt
    assert '"ranked_item_ids"' in request.prompt
    assert "Directed transition evidence" in request.prompt

