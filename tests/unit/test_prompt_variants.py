from llm4rec.prompts.base import PromptExample
from llm4rec.prompts.builder import build_prompt
from llm4rec.prompts.variants import PROMPT_VARIANTS


def test_all_phase3a_prompt_variants_are_deterministic_and_candidate_aware():
    example = PromptExample(
        user_id="u1",
        history=["i1", "i2"],
        target_item="i3",
        candidate_items=["i3", "i4"],
        item_records={
            "i1": {"item_id": "i1", "title": "One"},
            "i2": {"item_id": "i2", "title": "Two"},
            "i3": {"item_id": "i3", "title": "Three"},
            "i4": {"item_id": "i4", "title": "Four"},
        },
        history_rows=[
            {"user_id": "u1", "item_id": "i1", "timestamp": 0},
            {"user_id": "u1", "item_id": "i2", "timestamp": 3600},
        ],
        transition_evidence=[{"source_item": "i2", "target_item": "i3", "transition_count": 2}],
        time_window_evidence=[{"source_item": "i2", "target_item": "i3", "time_window_score_1d": 1.0}],
        contrastive_evidence=[{"source_item": "i2", "target_item": "i3", "text_similarity": 0.0}],
    )
    for name in PROMPT_VARIANTS:
        first = build_prompt(example, prompt_variant=name)
        second = build_prompt(example, prompt_variant=name)
        assert first.prompt == second.prompt
        assert "i3: Three" in first.prompt
        assert "i4: Four" in first.prompt
        assert "strict JSON" in first.prompt
        assert first.prompt_version.startswith("phase3a.")

