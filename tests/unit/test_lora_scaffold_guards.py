import pytest

from llm4rec.evaluation.lora_rerank import _guard_scaffold_eval
from llm4rec.trainers.lora_sft import _guard_scaffold_training


def test_lora_training_guard_rejects_reference_scaffold_by_default():
    config = {
        "baseline_contract": {"scaffold_only": True},
        "sft": {"scaffold_only": True},
    }

    with pytest.raises(RuntimeError, match="scaffold reference baseline"):
        _guard_scaffold_training(config, allow_scaffold=False)

    _guard_scaffold_training(config, allow_scaffold=True)


def test_lora_eval_guard_rejects_reference_scaffold_by_default():
    contract = {"scaffold_only": True}

    with pytest.raises(RuntimeError, match="scaffold reference baseline"):
        _guard_scaffold_eval(contract, allow_scaffold=False)

    _guard_scaffold_eval(contract, allow_scaffold=True)
