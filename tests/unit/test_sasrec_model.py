import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for SASRec model tests")

from llm4rec.models.sasrec import SASRecModel, causal_attention_mask


def test_sasrec_forward_shape():
    model = SASRecModel(num_items=5, hidden_dim=8, num_layers=1, num_heads=1, dropout=0.0, max_seq_len=4)
    seq = torch.tensor([[0, 1, 2, 3], [0, 0, 2, 4]], dtype=torch.long)
    output = model(seq)
    assert output.shape == (2, 4, 8)
    scores = model.score_items(seq, torch.tensor([[1, 2], [3, 4]], dtype=torch.long))
    assert scores.shape == (2, 2)


def test_causal_mask_blocks_future_positions():
    mask = causal_attention_mask(4)
    assert mask[0, 1].item() is True
    assert mask[1, 0].item() is False
    assert mask.diag().any().item() is False
