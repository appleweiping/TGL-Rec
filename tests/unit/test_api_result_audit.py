from llm4rec.diagnostics.api_result_audit import audit_api_micro_response
from llm4rec.llm.base import LLMResponse


def test_api_result_audit_tracks_parse_grounding_and_hallucinations():
    sample = {
        "candidate_items": ["i2", "i3"],
        "case_group": "transition_only",
        "domain": "tiny",
        "evidence_source_item": "i1",
        "evidence_target_item": "i2",
        "history": ["i1"],
        "sample_id": "s1",
        "target_item": "i3",
        "user_id": "u1",
    }
    response = LLMResponse(
        raw_output=(
            '{"ranked_item_ids":["i2","i999"],"reasoning_summary":"short",'
            '"evidence_used":[{"type":"transition","source_item":"i1","target_item":"i2","text":"edge"}]}'
        ),
        provider="openai_compatible",
        model="test-model",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        latency_ms=12.0,
    )
    prediction, parse_failure, hallucination = audit_api_micro_response(
        sample=sample,
        prompt_variant="history_with_transition_evidence",
        prompt_version="v1",
        response=response,
        transition_edges={("i1", "i2"): {"source_item": "i1", "target_item": "i2"}},
        time_window_edges={},
        time_bucket_by_pair={},
        run_mode="diagnostic_api",
    )
    assert parse_failure is None
    assert hallucination is not None
    assert prediction["metadata"]["parse_success"] is True
    assert prediction["metadata"]["grounding"]["evidence_grounding_rate"] == 1.0
    assert prediction["metadata"]["invalid_item_ids"] == ["i999"]
