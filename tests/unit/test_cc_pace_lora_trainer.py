"""Tests for the CC-PACE LoRA training driver's CPU-testable pieces."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from llm4rec.trainers.cc_pace_trainer import distance_correlation

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
try:
    from train_cc_pace_lora import build_user_prefix_sequences, torch_dcor
finally:
    sys.path.remove(str(_SCRIPTS))


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_torch_dcor_matches_numpy_reference(seed):
    rng = random.Random(seed)
    a = np.array([rng.gauss(0, 1) for _ in range(40)])
    b = 0.6 * a + np.array([rng.gauss(0, 0.5) for _ in range(40)])
    ref = distance_correlation(a, b)
    got = float(torch_dcor(torch.tensor(a), torch.tensor(b)))
    assert abs(got - ref) < 1e-8


def test_torch_dcor_is_differentiable_and_low_for_independent():
    rng = random.Random(7)
    a = torch.tensor([rng.gauss(0, 1) for _ in range(60)], requires_grad=True)
    b = torch.tensor([rng.gauss(0, 1) for _ in range(60)])
    d = torch_dcor(a, b)
    d.backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    assert float(d) < 0.45  # independent samples -> small dCor


def test_build_user_prefix_sequences_orders_by_timestamp():
    rows = [
        {"user_id": "u1", "item_id": "b", "timestamp": 2.0},
        {"user_id": "u1", "item_id": "a", "timestamp": 1.0},
        {"user_id": "u1", "item_id": "c", "timestamp": 3.0},
        {"user_id": "u2", "item_id": "x", "timestamp": 9.0},
    ]
    seqs = build_user_prefix_sequences(rows)
    assert seqs["u1"] == ["a", "b", "c"]
    assert seqs["u2"] == ["x"]
