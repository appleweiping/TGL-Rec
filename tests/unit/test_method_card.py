from llm4rec.methods.config import load_method_config
from llm4rec.methods.method_card import render_method_card


def test_method_card_documents_status_without_superiority_claim():
    card = render_method_card(load_method_config("configs/methods/time_graph_evidence.yaml"))
    assert "TimeGraphEvidenceRec" in card
    assert "Current reportable status: `false`" in card
    assert "No empirical superiority is claimed" in card
    assert "outperforms" not in card.lower()
