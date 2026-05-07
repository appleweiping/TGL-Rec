from __future__ import annotations

import pytest

from llm4rec.baselines.reference_methods import (
    ReferenceBaselineNotImplementedError,
    get_reference_method,
    reference_method_names,
    require_implemented_reference_method,
)


def test_reference_method_registry_selects_concrete_methods() -> None:
    names = reference_method_names()

    assert "slmrec_distill_qwen_lora" in names
    assert "llm_esr_qwen_lora" in names
    assert "controllable_rec_qwen_lora" in names
    assert "cllm4rec_qwen_lora" in names
    assert "rlmrec_qwen_lora" in names
    assert "transrec_qwen_lora" in names
    assert "review_pref_reasoning_qwen_lora" in names
    assert len(names) == 7


def test_reference_methods_are_not_reportable_until_implemented() -> None:
    spec = get_reference_method("cllm4rec_qwen_lora")

    assert spec.reportable_baseline is False
    assert spec.implementation_status == "not_implemented"
    assert "item prediction head" in spec.preserve_components
    with pytest.raises(ReferenceBaselineNotImplementedError):
        require_implemented_reference_method("cllm4rec_qwen_lora")
