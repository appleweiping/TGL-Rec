"""Equivalence test for the KV-cached batched label scorer in hf_judge.

The batched shared-prefill implementation MUST produce the same length-normalized
label log-probs as a naive per-label full forward — this is reporting-critical
scoring code. Runs on CPU with a tiny in-memory Llama-architecture model (no
download) and a deterministic char-level tokenizer; skips without torch/transformers.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from llm4rec.methods.cc_pace.hf_judge import score_label_logprobs  # noqa: E402


class CharTokenizer:
    """Minimal char-level tokenizer exposing the surface hf_judge uses."""

    pad_token_id = 0

    def __call__(self, text, return_tensors="pt", add_special_tokens=True,
                 truncation=False, max_length=None):
        del return_tensors, add_special_tokens
        ids = [min(ord(c), 255) + 1 for c in text]
        if truncation and max_length is not None:
            ids = ids[-max_length:]

        class _Enc:
            pass

        enc = _Enc()
        enc.input_ids = torch.tensor([ids], dtype=torch.long)
        return enc


def _tiny_model():
    cfg = transformers.LlamaConfig(
        vocab_size=300, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512,
    )
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(cfg).eval()


def _naive_reference(model, tok, prompt, label_ids):
    """Per-label full forward (the original implementation), as ground truth."""
    p_ids = tok(prompt).input_ids
    out = []
    with torch.no_grad():
        for label in label_ids:
            t_ids = tok(label, add_special_tokens=False).input_ids
            inp = torch.cat([p_ids, t_ids], dim=1)
            logits = model(inp).logits
            t_len = t_ids.shape[1]
            logp = torch.log_softmax(logits[0, -t_len - 1 : -1, :].float(), dim=-1)
            tgt = inp[0, -t_len:]
            tok_lp = logp[torch.arange(t_len), tgt]
            out.append(float(tok_lp.sum().item()) / t_len)
    return out


@pytest.mark.parametrize("suffix_batch", [128, 3])
def test_batched_scorer_matches_naive_full_forward(suffix_batch):
    model = _tiny_model()
    tok = CharTokenizer()
    prompt = "User bought serum and shampoo.\nCandidates:\n[000] serum\n[001] brush\nBest label:"
    labels = [f"[{i:03d}]" for i in range(7)]

    fast = score_label_logprobs(
        model, tok, prompt, labels, device="cpu", max_ctx=256, suffix_batch=suffix_batch
    )
    ref = _naive_reference(model, tok, prompt, labels)
    assert len(fast) == len(ref) == 7
    for f, r in zip(fast, ref):
        assert abs(f - r) < 1e-4, (f, r)


def test_batched_scorer_ranks_consistently_with_variable_label_lengths():
    model = _tiny_model()
    tok = CharTokenizer()
    prompt = "History: lipstick, mascara.\nBest label:"
    labels = ["[01]", "[002]", "[3]", "[0004]"]  # heterogeneous token lengths
    fast = score_label_logprobs(model, tok, prompt, labels, device="cpu", max_ctx=256)
    ref = _naive_reference(model, tok, prompt, labels)
    order_fast = sorted(range(len(labels)), key=lambda i: -fast[i])
    order_ref = sorted(range(len(labels)), key=lambda i: -ref[i])
    assert order_fast == order_ref
    for f, r in zip(fast, ref):
        assert abs(f - r) < 1e-4
