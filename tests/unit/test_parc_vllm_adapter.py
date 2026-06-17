"""CPU unit tests for the PaRC vLLM duel adapter (import-safety + batch wiring).

These tests run on a host with NO vllm installed: they assert the module imports
cleanly, the offline CPU fallback engages, ``duel_logits_batch`` runs ONE pass
over a prompt list, the offline logit matches the protocol sign convention
(>0 favours slot-A), and the adapter satisfies the ``PairwiseDuelModel`` protocol.
"""

from __future__ import annotations

import importlib

from llm4rec.methods.parc.config import PaRCConfig
from llm4rec.methods.parc.pairwise_prompt import (
    PairwiseDuelModel,
    build_pairwise_prompt,
)


def test_module_imports_without_vllm():
    """The adapter module must import on CPU with vllm absent."""
    mod = importlib.import_module("llm4rec.methods.parc.vllm_duel_model")
    assert hasattr(mod, "VLLMDuelModel")
    # _vllm_available must not raise and must not import vllm eagerly.
    assert isinstance(mod._vllm_available(), bool)


def test_offline_construction_engages_fallback():
    from llm4rec.methods.parc.vllm_duel_model import VLLMDuelModel

    m = VLLMDuelModel(model="dummy/path", offline=True)
    assert m.offline is True
    assert m._llm is None
    assert "offline" in m.extraction.lower()
    assert "not_for_reporting" in m.extraction.lower()


def test_satisfies_pairwise_duel_model_protocol():
    from llm4rec.methods.parc.vllm_duel_model import VLLMDuelModel

    m = VLLMDuelModel(offline=True)
    assert isinstance(m, PairwiseDuelModel)


def test_batch_adapter_one_pass_and_length():
    """duel_logits_batch returns one logit per prompt, in order."""
    from llm4rec.methods.parc.vllm_duel_model import VLLMDuelModel

    cfg = PaRCConfig()
    m = VLLMDuelModel(offline=True)
    history = ["alpha widget", "beta gadget"]
    prompts = [
        build_pairwise_prompt(history, "alpha widget", "zzz nothing", cfg),
        build_pairwise_prompt(history, "zzz nothing", "beta gadget", cfg),
    ]
    out = m.duel_logits_batch(prompts)
    assert len(out) == len(prompts)
    assert all(isinstance(x, float) for x in out)


def test_offline_logit_sign_convention():
    """Offline logit > 0 when slot-A overlaps history more than slot-B."""
    from llm4rec.methods.parc.vllm_duel_model import VLLMDuelModel

    cfg = PaRCConfig()
    m = VLLMDuelModel(offline=True)
    history = ["red apple juice", "green apple"]
    # A overlaps history ("apple"), B does not.
    p = build_pairwise_prompt(history, "fresh apple", "metal hammer", cfg)
    assert m.duel_logit(p) > 0
    # swap: now B overlaps, A does not -> negative.
    p2 = build_pairwise_prompt(history, "metal hammer", "fresh apple", cfg)
    assert m.duel_logit(p2) < 0


def test_offline_batch_matches_single():
    from llm4rec.methods.parc.vllm_duel_model import VLLMDuelModel

    cfg = PaRCConfig()
    m = VLLMDuelModel(offline=True)
    history = ["foo bar", "baz"]
    prompts = [
        build_pairwise_prompt(history, "foo thing", "qux thing", cfg),
        build_pairwise_prompt(history, "qux thing", "bar thing", cfg),
    ]
    batched = m.duel_logits_batch(prompts)
    singles = [m.duel_logit(p) for p in prompts]
    assert batched == singles


def test_empty_batch_returns_empty():
    from llm4rec.methods.parc.vllm_duel_model import VLLMDuelModel

    m = VLLMDuelModel(offline=True)
    assert m.duel_logits_batch([]) == []
