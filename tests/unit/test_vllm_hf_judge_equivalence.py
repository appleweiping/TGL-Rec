"""GPU-only equivalence tests for HF and vLLM CC-PACE judges.

These tests are intended for the server vLLM environment. They skip on CPU CI
or developer machines without vLLM/CUDA/model weights.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


LABELS = ["[000]", "[001]", "[010]", "[042]", "[100]"]

PANELS = [
    """You are given a user's recent purchases and 5 candidate products.

User recent purchases (oldest to newest): gentle cleanser; vitamin c serum

Candidates:
[000]
category: Beauty > Skincare
brand: Aster
keywords: brightening vitamin c serum
attrs: fragrance free, daytime
[001]
category: Beauty > Hair
brand: Bristle
keywords: volumizing dry shampoo
attrs: travel size
[010]
category: Beauty > Skincare
brand: Calma
keywords: barrier repair moisturizer
attrs: ceramide, sensitive skin
[042]
category: Beauty > Makeup
brand: Dusk
keywords: matte lip color
attrs: rose shade
[100]
category: Beauty > Tools
brand: Edge
keywords: facial cleansing brush
attrs: silicone

Best label:""",
    """You are given a user's recent purchases and 5 candidate products.

User recent purchases (oldest to newest): SPF moisturizer; after-sun aloe gel
User profile:
concerns: dryness; sun protection
liked_brands: coastal lab

Candidates:
[000]
category: Beauty > Skincare
brand: Coastal Lab
keywords: mineral sunscreen lotion
attrs: SPF 50, hydrating
[001]
category: Beauty > Fragrance
brand: Amber House
keywords: warm vanilla perfume
attrs: eau de parfum
[010]
category: Beauty > Skincare
brand: North Star
keywords: retinol night cream
attrs: advanced, unscented
[042]
category: Beauty > Makeup
brand: Linework
keywords: waterproof eyeliner
attrs: black
[100]
category: Beauty > Bath
brand: Salt & Stone
keywords: exfoliating body scrub
attrs: citrus

Best label:""",
]


def test_vllm_matches_hf_label_logprobs_on_synthetic_panels():
    pytest.importorskip("vllm")
    pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the vLLM/HF judge equivalence test")

    model_path = os.environ.get("CC_PACE_JUDGE_MODEL", "/home/ajifang/models/Qwen/Qwen3-8B")
    if not Path(model_path).exists():
        pytest.skip(f"judge model path is not available: {model_path}")

    from llm4rec.methods.cc_pace.hf_judge import HFForcedChoiceModel
    from llm4rec.methods.cc_pace.vllm_judge import VLLMForcedChoiceModel

    max_ctx = int(os.environ.get("CC_PACE_EQUIV_MAX_CTX", "1024"))
    gpu_memory_utilization = float(os.environ.get("CC_PACE_VLLM_GPU_MEMORY_UTILIZATION", "0.35"))

    vllm_model = VLLMForcedChoiceModel(
        model_path,
        max_ctx=max_ctx,
        max_model_len=max_ctx + 16,
        max_num_seqs=len(LABELS),
        gpu_memory_utilization=gpu_memory_utilization,
    )
    hf_model = HFForcedChoiceModel(
        model_path,
        max_ctx=max_ctx,
        suffix_batch=len(LABELS),
    )

    try:
        for prompt in PANELS:
            hf_scores = np.array(hf_model.label_logprobs(prompt, LABELS), dtype=float)
            vllm_scores = np.array(vllm_model.label_logprobs(prompt, LABELS), dtype=float)
            np.testing.assert_allclose(vllm_scores, hf_scores, atol=1e-2, rtol=0.0)
    finally:
        del hf_model
        del vllm_model
        torch.cuda.empty_cache()
