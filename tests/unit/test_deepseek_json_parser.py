import pytest

from llm4rec.llm.json_parser import LLMJSONParseError, parse_rerank_json


def test_parse_rerank_json_filters_invalid_and_duplicates():
    parsed = parse_rerank_json(
        '{"ranked_item_ids":["i1","bad","i1","i2"],'
        '"evidence_usage":{"transition":true,"time":false,"semantic":true,"contrastive":false}}',
        candidate_item_ids=["i1", "i2"],
    )

    assert parsed.ranked_item_ids == ["i1", "i2"]
    assert parsed.invalid_item_ids == ["bad"]
    assert parsed.duplicate_item_ids == ["i1"]
    assert parsed.evidence_usage["transition"] is True


def test_parse_rerank_json_requires_ranked_item_ids():
    with pytest.raises(LLMJSONParseError):
        parse_rerank_json('{"items":["i1"]}', candidate_item_ids=["i1"])
