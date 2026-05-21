"""Quick smoke test for new TGL-Rec modules."""
import sys
sys.path.insert(0, "src")

from llm4rec.evidence.need_state import NeedStateEncoder, NeedState
from llm4rec.evidence.need_gate import LearnedNeedGate, GateConfig
from llm4rec.evidence.reportable_scorer import ReportableScorer

# Test NeedStateEncoder
encoder = NeedStateEncoder(
    recent_window=3,
    item_categories={"a": "cat1", "b": "cat1", "c": "cat2", "d": "cat2", "e": "cat3"}
)
state = encoder.encode(
    history=["a", "b", "c", "d", "e"],
    timestamps=[1.0, 2.0, 3.0, 4.0, 5.0],
    prediction_timestamp=10.0,
)
print(f"NeedState: drift={state.drift_magnitude:.3f}, pressure={state.transition_pressure:.3f}, "
      f"gap={state.temporal_gap:.3f}, entropy={state.category_entropy:.3f}")
assert len(state.to_vector()) == 5, "Need-state vector should have 5 dims"

# Test LearnedNeedGate (untrained fallback)
gate = LearnedNeedGate()
alpha = gate.predict(
    [0.5, 0.8, 0.1, 0.7, 0.2],
    [1.0, 0.5, 0.3, 0.2, 0.1, 0.4, 0.2, 0.0, 0.3, 0.6],
)
print(f"Gate alpha (untrained fallback): {alpha:.3f}")
assert 0 <= alpha <= 1

# Test gate training
examples = [
    {"need_state": [0.8, 0.9, 0.1, 0.7, 0.5],
     "evidence_features": [2.0, 0.5, 1.0, 0.3, 0.2, 0.5, 0.1, 0.0, 0.3, 0.8],
     "label": 1, "weight": 1.0},
    {"need_state": [0.1, 0.1, 0.9, 0.2, 0.8],
     "evidence_features": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.1, 0.1],
     "label": 0, "weight": 1.0},
] * 50

metrics = gate.train(examples)
print(f"Gate training metrics: epochs={metrics['epochs_trained']}, loss={metrics['final_train_loss']:.4f}")
assert gate.weights.trained, "Gate should be marked as trained"

alpha_pos = gate.predict(
    [0.8, 0.9, 0.1, 0.7, 0.5],
    [2.0, 0.5, 1.0, 0.3, 0.2, 0.5, 0.1, 0.0, 0.3, 0.8],
)
alpha_neg = gate.predict(
    [0.1, 0.1, 0.9, 0.2, 0.8],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.1, 0.1],
)
print(f"Gate alpha (trained): temporal_rich={alpha_pos:.3f}, semantic_only={alpha_neg:.3f}")
assert alpha_pos > alpha_neg, (
    f"Gate should give higher alpha to temporal-rich examples: {alpha_pos} vs {alpha_neg}"
)

# Test ReportableScorer
scorer = ReportableScorer(gate=gate, need_state_encoder=encoder)
state = scorer.set_user_context(
    history=["a", "b", "c", "d", "e"],
    timestamps=[1.0, 2.0, 3.0, 4.0, 5.0],
    prediction_timestamp=10.0,
)
print(f"Scorer context set: {state.to_dict()}")

print("\n=== ALL TESTS PASSED ===")
