import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for TemporalGraphEncoder tests")

from llm4rec.encoders.temporal_graph_encoder import TemporalGraphEncoder


def test_temporal_graph_update_changes_user_memory():
    encoder = TemporalGraphEncoder(
        num_users=1,
        num_items=1,
        hidden_dim=8,
        user_to_idx={"u1": 1},
        item_to_idx={"i1": 1},
    )
    before = torch.tensor(encoder.encode_user("u1", 1))
    encoder.update({"user_id": "u1", "item_id": "i1", "timestamp": 1})
    after = torch.tensor(encoder.encode_user("u1", 1))
    assert not torch.allclose(before, after)


def test_temporal_graph_save_load_consistency(tmp_path):
    encoder = TemporalGraphEncoder(
        num_users=1,
        num_items=1,
        hidden_dim=8,
        user_to_idx={"u1": 1},
        item_to_idx={"i1": 1},
    )
    encoder.update({"user_id": "u1", "item_id": "i1", "timestamp": 1})
    path = tmp_path / "encoder.pt"
    encoder.save(path)
    loaded = TemporalGraphEncoder.load(path)
    assert loaded.score("u1", "i1", 1) == pytest.approx(encoder.score("u1", "i1", 1))
