"""Baseline rankers for the llm4rec experimental skeleton."""

from llm4rec.rankers.base import BaseRanker, RankingExample, RankingResult
from llm4rec.rankers.bm25 import BM25Ranker
from llm4rec.rankers.cc_pace import CCPaceRanker
from llm4rec.rankers.mf import MatrixFactorizationRanker
from llm4rec.rankers.popularity import PopularityRanker
from llm4rec.rankers.random import RandomRanker

__all__ = [
    "BM25Ranker",
    "BaseRanker",
    "CCPaceRanker",
    "MatrixFactorizationRanker",
    "PaRCRanker",
    "PopularityRanker",
    "RandomRanker",
    "RankingExample",
    "RankingResult",
]


def __getattr__(name: str):
    # Lazy export for PaRCRanker: PaRC's ranker imports rankers.base (which runs
    # this package __init__), so an EAGER `from .parc import PaRCRanker` here
    # creates a circular import whenever methods.parc.ranker is imported directly
    # (tests, M0 driver). Loading it on first attribute access avoids the cycle
    # while keeping `from llm4rec.rankers import PaRCRanker` working.
    if name == "PaRCRanker":
        from llm4rec.rankers.parc import PaRCRanker
        return PaRCRanker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
