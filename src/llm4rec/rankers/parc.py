"""Registry shim: expose PaRCRanker in the rankers namespace.

PaRC (Pairwise-Relational Calibration) lives in ``llm4rec.methods.parc``; this
mirrors ``rankers/cc_pace.py`` so the evaluation harness can select it by the
``rankers`` namespace like every other ranker.
"""

from llm4rec.methods.parc.ranker import PaRCRanker

__all__ = ["PaRCRanker"]
