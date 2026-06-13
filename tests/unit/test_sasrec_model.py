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


def test_final_state_nonzero_for_left_padded_sequence():
    """Regression (left/right-pad mismatch): final_state must read the true last
    item for LEFT-padded sequences (real items at the END, padding at the FRONT).

    The old implementation gathered ``ne(pad).sum()-1`` (= count-1), which is only
    correct for RIGHT-padding; for left-padding it lands inside the front padding
    region, which ``masked_fill`` zeroes, collapsing the state -- and therefore all
    candidate scores -- to 0. This zeroed state is what made the SASRec smoke loss
    stick at -log sigma(0)=0.69 and produced constant predictions / degenerate CF
    artifacts.
    """
    torch.manual_seed(0)
    model = SASRecModel(num_items=6, hidden_dim=8, num_layers=1, num_heads=1, dropout=0.0, max_seq_len=4)
    model.eval()
    left = torch.tensor([[0, 0, 2, 4]], dtype=torch.long)  # real items 2,4 at the end
    state = model.final_state(left)
    assert state.shape == (1, 8)
    assert state.abs().sum().item() > 0.0, "final_state collapsed to zero for a left-padded sequence"


def test_final_state_selects_last_real_item_both_paddings():
    """final_state must return the encoded state at the last NON-padding position
    regardless of left- vs right-padding, so left-padded trainers and the
    right-padded CF builder (commit 754e13c) both score correctly."""
    torch.manual_seed(0)
    model = SASRecModel(num_items=6, hidden_dim=8, num_layers=1, num_heads=1, dropout=0.0, max_seq_len=4)
    model.eval()
    left = torch.tensor([[0, 0, 2, 4]], dtype=torch.long)   # last real at index 3
    right = torch.tensor([[2, 4, 0, 0]], dtype=torch.long)  # last real at index 1
    enc_left = model(left)
    enc_right = model(right)
    assert torch.allclose(model.final_state(left), enc_left[:, 3, :], atol=1e-6)
    assert torch.allclose(model.final_state(right), enc_right[:, 1, :], atol=1e-6)
    assert model.final_state(right).abs().sum().item() > 0.0
