"""CC-PACE ranker: BaseRanker-compatible entry point for the eval harness.

Wraps the CC-PACE statistic pipeline (methods/cc_pace) behind the same
``fit`` / ``rank`` / ``save_artifact`` contract as every other ranker, so it
drops into the existing evaluation harness, comparison tables, and ablation
runner without special-casing.

Zero-shot (frozen judge, no LoRA) is the default and is what the beauty kill
test runs. A trained judge is supplied by passing an already-LoRA-adapted
``EvidenceModel`` (see llm4rec.trainers.cc_pace_trainer); the ranker itself is
inference-only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.io.artifacts import write_json
from llm4rec.methods.cc_pace import statistic as stat_mod
from llm4rec.methods.cc_pace.cf_conditioning import CFProvider, NullCFProvider
from llm4rec.methods.cc_pace.config import CCPaceConfig, DEFAULT_CC_PACE_CONFIG
from llm4rec.methods.cc_pace.judge import EvidenceModel
from llm4rec.rankers.base import RankingExample, RankingResult


class _MockEvidenceModel:
    """Deterministic CPU stand-in used when no real judge is provided.

    Scores a candidate by token overlap between its rendered block and the user
    block. Lets the harness/tests run end-to-end without an 8B model; NOT for
    reporting. Replace with HFForcedChoiceModel on the server.
    """

    def label_logprobs(self, prompt: str, label_ids: list[str]) -> list[float]:
        blocks = prompt.split("Candidates:", 1)[-1]
        user = prompt.split("Candidates:", 1)[0].lower()
        user_toks = set(user.split())
        out = []
        for label in label_ids:
            seg = blocks.split(label, 1)[-1]
            seg = seg.split("[", 1)[0]
            toks = set(seg.lower().split())
            overlap = len(toks & user_toks)
            out.append(float(overlap) / (len(toks) + 1.0))
        return out


class CCPaceRanker:
    """Collaborative-Conditioned Panel-Anomaly Calibrated Exchangeability ranker."""

    name = "cc_pace"

    def __init__(
        self,
        config: CCPaceConfig | None = None,
        *,
        model: EvidenceModel | None = None,
        cf_provider: CFProvider | None = None,
        seed: int = 0,
    ) -> None:
        self.config = config or DEFAULT_CC_PACE_CONFIG
        self.model = model or _MockEvidenceModel()
        self.cf_provider = cf_provider or NullCFProvider()
        self.seed = seed
        self._profiles: dict[str, dict[str, Any]] = {}
        self._item_meta: dict[str, dict[str, Any]] = {}

    def fit(
        self,
        train_interactions: list[dict[str, Any]],
        item_records: list[dict[str, Any]],
    ) -> None:
        """CC-PACE judge is frozen/zero-shot here; we only index item metadata.

        Training the judge LoRA is a separate offline step
        (llm4rec.trainers.cc_pace_trainer); the resulting adapter is loaded into
        ``model`` before ranking. ``fit`` indexes item records so candidate dicts
        can be enriched with schema fields + popularity at rank time.
        """
        self._item_meta = {str(it["item_id"]): dict(it) for it in item_records}

    def set_profiles(self, profiles: dict[str, dict[str, Any]]) -> None:
        """Attach history-derived long-term profile slots, keyed by user_id."""
        self._profiles = dict(profiles)

    def _candidate_dicts(self, example: RankingExample) -> list[dict[str, Any]]:
        out = []
        for cid in example.candidate_items:
            meta = self._item_meta.get(str(cid), {})
            out.append(
                {
                    "item_id": str(cid),
                    "category": meta.get("category", meta.get("categories", "")),
                    "brand": meta.get("brand", ""),
                    "keywords": meta.get("keywords", meta.get("title", "")),
                    "attrs": meta.get("attrs", meta.get("attributes", "")),
                    "popularity": meta.get("popularity", meta.get("pop_count", 0.0)),
                }
            )
        return out

    def rank(self, example: RankingExample) -> RankingResult:
        candidates = self._candidate_dicts(example)
        candidate_ids = [str(c) for c in example.candidate_items]
        result = stat_mod.compute_statistic(
            user_id=example.user_id,
            candidates=candidates,
            candidate_ids=candidate_ids,
            history_titles=list(example.history),
            profile=self._profiles.get(example.user_id),
            model=self.model,
            cf_provider=self.cf_provider,
            cfg=self.config,
            seed=self.seed,
            n_history=len(example.history),
        )
        t = result["T"]
        order = sorted(range(len(candidate_ids)), key=lambda i: (-float(t[i]), candidate_ids[i]))
        return RankingResult(
            user_id=example.user_id,
            items=[candidate_ids[i] for i in order],
            scores=[float(t[i]) for i in order],
            raw_output=None,
            metadata={
                "method": self.name,
                "p_pop": [float(result["p_pop"][i]) for i in order],
                "uncertainty": [float(result["uncertainty"][i]) for i in order],
                "use_cf_tokens": self.config.use_cf_tokens,
                "use_residualizer": self.config.use_residualizer,
            },
        )

    def save_artifact(self, output_dir: str | Path) -> None:
        write_json(Path(output_dir) / "cc_pace_config.json", self.config.to_dict())
