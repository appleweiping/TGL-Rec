import pytest

from llm4rec.evidence.temporal_graph import make_semantic_evidence
from llm4rec.methods.leakage import LeakageValidator, LeakageViolation


def test_leakage_validator_blocks_target_in_history():
    validator = LeakageValidator(reportable=False)
    with pytest.raises(LeakageViolation, match="target item"):
        validator.validate_example(history=["i1", "i2"], target_item="i2", candidate_items=["i2"])


def test_leakage_validator_blocks_diagnostic_evidence_for_reportable_run():
    evidence = make_semantic_evidence(
        source_item="i1",
        target_item="i2",
        similarity=0.2,
        candidate_protocol="full_catalog",
        constructed_from="diagnostic_only",
    )
    validator = LeakageValidator(reportable=True)
    with pytest.raises(LeakageViolation, match="train_only"):
        validator.validate_evidence(evidence)


def test_leakage_validator_blocks_unsafe_reportable_config_modes():
    validator = LeakageValidator(reportable=True)
    for config in (
        {"method": {"name": "time_graph_evidence_rec", "reportable": True}, "llm": {"provider": "mock"}},
        {"method": {"name": "time_graph_evidence_rec", "reportable": True}, "encoder": {"type": "temporal_memory_stub"}},
        {"method": {"name": "skeleton_smoke", "reportable": True}},
    ):
        with pytest.raises(LeakageViolation):
            validator.validate_config(config)


def test_leakage_validator_blocks_prompt_target_label():
    validator = LeakageValidator(reportable=False)
    with pytest.raises(LeakageViolation, match="target label"):
        validator.validate_prompt(prompt="Rank candidates. Correct target is i9.", target_item="i9")
