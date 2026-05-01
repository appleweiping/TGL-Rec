import pytest

from llm4rec.rankers.base import RankingExample
from llm4rec.rankers.sequential import MarkovTransitionRanker, SASRecInterface


def test_markov_transition_ranker_uses_last_history_item():
    ranker = MarkovTransitionRanker()
    train = [
        {"user_id": "u1", "item_id": "i1", "timestamp": 1},
        {"user_id": "u1", "item_id": "i2", "timestamp": 2},
        {"user_id": "u2", "item_id": "i1", "timestamp": 1},
        {"user_id": "u2", "item_id": "i2", "timestamp": 2},
    ]
    items = [{"item_id": "i1"}, {"item_id": "i2"}, {"item_id": "i3"}]
    ranker.fit(train, items)
    result = ranker.rank(
        RankingExample(user_id="u3", history=["i1"], target_item="i2", candidate_items=["i3", "i2"])
    )
    assert result.items[0] == "i2"
    assert result.metadata["reportable"] is False


def test_sasrec_interface_is_explicitly_not_implemented():
    with pytest.raises(NotImplementedError):
        SASRecInterface().fit([], [])
