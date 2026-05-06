from llm4rec.evaluation.lora_rerank import _limit_candidates


def test_limit_candidates_samples_negatives_and_preserves_target_position():
    candidates = [f"i{index}" for index in range(100)]
    target = "i99"

    limited = _limit_candidates(candidates, target=target, limit=10, seed_key="u1|i99")
    repeated = _limit_candidates(candidates, target=target, limit=10, seed_key="u1|i99")

    assert limited == repeated
    assert len(limited) == 10
    assert target in limited
    assert limited != candidates[:9] + [target]
    assert limited[-1] != target
