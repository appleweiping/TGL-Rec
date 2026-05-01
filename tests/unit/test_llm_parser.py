import pytest

from llm4rec.prompts.parsers import LLMParseError, parse_llm_response


def test_parser_handles_fenced_json_and_hallucinated_ids():
    raw = """Here:
```json
{"ranked_item_ids":["i2","bad","i2","i1"],"reasoning_summary":"ok","evidence_used":[]}
```
done"""
    parsed = parse_llm_response(raw, candidate_items=["i1", "i2"])

    assert parsed.ranked_item_ids == ["i2", "i1"]
    assert parsed.invalid_item_ids == ["bad"]
    assert parsed.duplicate_item_ids == ["i2"]
    assert parsed.reasoning_summary == "ok"


def test_parser_rejects_unrecoverable_non_json():
    with pytest.raises(LLMParseError):
        parse_llm_response("not json", candidate_items=["i1"])

