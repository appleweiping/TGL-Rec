import json

from llm4rec.evidence.base import Evidence
from llm4rec.evidence.translator import GraphToTextTranslator


def _evidence(metadata=None):
    return Evidence(
        evidence_id="ev_1",
        evidence_type="transition",
        source_item="i1",
        target_item="i2",
        support_items=["i1", "i2"],
        timestamp_info={"gap_bucket": "same_session", "mean_gap_seconds": 60},
        stats={"transition_count": 2},
        provenance={
            "graph_artifact": "transition_edges.jsonl",
            "split": "train",
            "candidate_protocol": "full_catalog",
            "constructed_from": "train_only",
        },
        metadata=metadata or {},
    )


def test_compact_translator_uses_only_evidence_fields():
    text = GraphToTextTranslator("compact").translate(_evidence())
    assert "i1" in text and "i2" in text
    assert "transition_count=2" in text
    assert "Imaginary" not in text


def test_prompt_ready_json_translator_is_strict_json():
    text = GraphToTextTranslator("prompt_ready_json").translate([_evidence({"source_title": "A"})])
    payload = json.loads(text)
    assert payload["evidence"][0]["metadata"]["source_title"] == "A"
