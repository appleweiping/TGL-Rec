from __future__ import annotations

from llm4rec.evidence.base import Evidence
from llm4rec.evidence.scorer import factorized_score_candidates


def _evidence(evidence_type: str, target: str, stats: dict) -> Evidence:
    return Evidence(
        evidence_id=f"ev_{evidence_type}_{target}",
        evidence_type=evidence_type,
        source_item="i_hist",
        target_item=target,
        support_items=["i_hist", target],
        stats=stats,
        provenance={
            "candidate_protocol": "fixed_shared_candidates",
            "constructed_from": "train_only",
            "graph_artifact": "unit",
            "split": "train",
        },
    )


def test_need_gate_prefers_temporal_transition_over_semantic_trap() -> None:
    evidence = [
        _evidence(
            "transition",
            "i_need",
            {"direction_asymmetry": 1.0, "lift": 3.0, "pmi": 1.0, "transition_count": 8, "transition_probability": 0.8},
        ),
        _evidence("contrastive", "i_need", {"transition_count": 8, "semantic_similarity": 0.05}),
        _evidence("semantic", "i_semantic", {"semantic_similarity": 0.9}),
    ]

    scores = factorized_score_candidates(
        evidence,
        ["i_need", "i_semantic"],
        {
            "contrastive_transition_weight": 0.5,
            "semantic_trap_penalty": 0.2,
            "semantic_weight": 1.0,
            "transition_weight": 1.0,
        },
    )

    assert scores["i_need"].total_score > scores["i_semantic"].total_score
    assert scores["i_need"].gate > scores["i_semantic"].gate
    assert scores["i_semantic"].semantic_trap_penalty > 0.0


def test_need_gate_exposes_factor_decomposition() -> None:
    evidence = [
        _evidence("transition", "i2", {"transition_count": 2}),
        _evidence("time_window", "i2", {"time_window_score": 0.25}),
        _evidence("history", "i2", {"recent_signal": 0.5}),
    ]

    score = factorized_score_candidates(evidence, ["i2"])["i2"].to_dict()

    assert score["evidence_counts"] == {"history": 1, "time_window": 1, "transition": 1}
    assert score["temporal_score"] > 0.0
    assert 0.0 <= score["gate"] <= 1.0
