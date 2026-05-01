from llm4rec.llm.structured_output import (
    EVIDENCE_TYPES,
    STRUCTURED_OUTPUT_SCHEMA_VERSION,
    api_micro_response_schema,
    openai_response_format,
)


def test_api_micro_schema_requires_strict_fields():
    schema = api_micro_response_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"ranked_item_ids", "reasoning_summary", "evidence_used"}
    evidence_schema = schema["properties"]["evidence_used"]["items"]
    assert set(evidence_schema["properties"]["type"]["enum"]) == set(EVIDENCE_TYPES)


def test_openai_response_format_can_be_disabled():
    assert openai_response_format(enabled=False) is None
    response_format = openai_response_format(enabled=True, strict=True)
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert STRUCTURED_OUTPUT_SCHEMA_VERSION.endswith(".v1")
