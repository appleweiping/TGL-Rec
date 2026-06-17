"""PaRC ranker: ties pony anchor + injectable pairwise duels -> re-ranked scores.

BaseRanker-compatible (``fit`` / ``rank`` / ``save_artifact``) so it drops into the
existing evaluation harness alongside ``CCPaceRanker``.

Pipeline per user:
  1. Look up the pony pointwise posterior for this user's candidates (the anchor).
     PaRC NEVER re-derives pony; it consumes existing pony scores.
  2. Run the adaptive O(K log K) duel schedule (``duels.run_duels``), evaluating
     each scheduled pair with the symmetrized A/B-swap duel (``pairwise_prompt``)
     via the INJECTED ``PairwiseDuelModel`` (mockable on CPU).
  3. Fit the anchored symmetrized BT field theta = alpha*pony + beta
     (``bt_field.fit_bt_field``).
  4. Final score = pony + lambda*beta (``calibrate.mixture_score``). lambda is a
     ranker attribute selected offline on validation (``calibrate.select_lambda``);
     lambda=0 recovers the pony order (the empirical floor).

The duel model is injected exactly like CC-PACE injects ``EvidenceModel``; a
deterministic ``_MockPairwiseDuelModel`` lets the harness/tests run end-to-end on
CPU with no 8B model. The server wires a concrete vLLM duel model reusing pony's
``run_ccrp_v3_domain_seeded`` pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from llm4rec.methods.parc import bt_field as bt_mod
from llm4rec.methods.parc import calibrate as calib_mod
from llm4rec.methods.parc import duels as duels_mod
from llm4rec.methods.parc import pairwise_prompt as pp_mod
from llm4rec.methods.parc.config import DEFAULT_PARC_CONFIG, PaRCConfig
from llm4rec.methods.parc.pairwise_prompt import PairwiseDuelModel
from llm4rec.rankers.base import RankingExample, RankingResult


class _MockPairwiseDuelModel:
    """Deterministic CPU stand-in: prefers the item whose title overlaps history.

    ``duel_logit`` parses the rendered A/B prompt and returns
    (overlap_A - overlap_B), where overlap_X = #(title tokens X ∩ history tokens).
    Symmetric by construction up to the swap, so A/B-swap symmetrization cancels
    nothing it shouldn't. NOT for reporting; replace with the vLLM duel model.
    """

    def duel_logit(self, prompt: str) -> float:
        body = prompt.split("Two candidate items:", 1)[-1]
        hist_block = prompt.split("Two candidate items:", 1)[0].lower()
        hist_toks = set(hist_block.replace("-", " ").split())
        a_seg = body.split("A:", 1)[-1].split("B:", 1)[0]
        b_seg = body.split("B:", 1)[-1].split("Given", 1)[0]
        a = len(set(a_seg.lower().split()) & hist_toks)
        b = len(set(b_seg.lower().split()) & hist_toks)
        return float(a - b)


class PaRCRanker:
    """Pairwise-Relational Calibration reranker (anchored comparative-residual)."""

    name = "parc"

    def __init__(
        self,
        config: PaRCConfig | None = None,
        *,
        model: PairwiseDuelModel | None = None,
        lam: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.config = config or DEFAULT_PARC_CONFIG
        self.model = model or _MockPairwiseDuelModel()
        self.lam = float(lam)            # selected offline on validation (0 = pony floor)
        self.seed = seed
        self._item_meta: dict[str, dict[str, Any]] = {}
        # user_id -> {item_id -> pony posterior}; loaded via set_pony_scores.
        self._pony: dict[str, dict[str, float]] = {}
        self.last_diagnostics: dict[str, Any] = {}

    def fit(
        self,
        train_interactions: list[dict[str, Any]],
        item_records: list[dict[str, Any]],
    ) -> None:
        """Index item metadata for rendering candidate titles in duel prompts.

        PaRC's duel model is frozen/zero-shot; there is no training step here. The
        pony anchor is supplied separately via ``set_pony_scores``.
        """
        self._item_meta = {str(it["item_id"]): dict(it) for it in item_records}

    def set_pony_scores(self, pony_scores: dict[str, dict[str, float]]) -> None:
        """Attach the existing pony per-candidate posteriors, keyed by user_id."""
        self._pony = {str(u): {str(c): float(s) for c, s in d.items()} for u, d in pony_scores.items()}

    def set_lambda(self, lam: float) -> None:
        """Set the validation-selected mixture weight (>=0; 0 recovers pony)."""
        self.lam = max(0.0, float(lam))

    def _candidate_titles(self, candidate_ids: list[str]) -> list[str]:
        out = []
        for cid in candidate_ids:
            meta = self._item_meta.get(str(cid), {})
            out.append(str(meta.get("title", meta.get("keywords", cid))))
        return out

    def _pony_vector(self, user_id: str, candidate_ids: list[str]) -> np.ndarray:
        d = self._pony.get(str(user_id), {})
        # missing pony score -> 0.0 (neutral anchor); harness guarantees coverage.
        return np.array([float(d.get(str(c), 0.0)) for c in candidate_ids], dtype=float)

    def rank(self, example: RankingExample) -> RankingResult:
        candidate_ids = [str(c) for c in example.candidate_items]
        titles = self._candidate_titles(candidate_ids)
        history = list(example.history)
        pony = self._pony_vector(example.user_id, candidate_ids)
        cfg = self.config

        # duel evaluator over ORIGINAL indices, via symmetrized A/B-swap duel.
        def eval_fn(i: int, j: int) -> float:
            return pp_mod.symmetrized_duel(history, titles[i], titles[j], self.model, cfg)["s_ij"]

        sched = duels_mod.run_duels(pony, eval_fn, cfg, seed=self.seed)
        bt = bt_mod.fit_bt_field(
            sched.as_pair_list(), pony,
            n_items=len(candidate_ids), l2_beta=cfg.bt_l2_beta,
            fit_alpha=cfg.bt_fit_alpha, anchor=cfg.anchor_pony,
        )
        scores = calib_mod.mixture_score(pony, bt.beta, self.lam, standardize_beta=cfg.standardize_beta)

        order = sorted(range(len(candidate_ids)), key=lambda i: (-float(scores[i]), candidate_ids[i]))
        self.last_diagnostics = {
            "n_comparisons": sched.n_comparisons,
            "duel_budget": sched.budget,
            "alpha": bt.alpha,
            "var_beta": bt.var_beta,
            "order_var_fraction_beta": bt.order_var_fraction_beta,
            "residual_rms": bt.residual_rms,
            "lambda": self.lam,
        }
        return RankingResult(
            user_id=example.user_id,
            items=[candidate_ids[i] for i in order],
            scores=[float(scores[i]) for i in order],
            raw_output=None,
            metadata={
                "method": self.name,
                "lambda": self.lam,
                "n_comparisons": sched.n_comparisons,
                "duel_budget": sched.budget,
                "alpha": float(bt.alpha),
                "var_beta": float(bt.var_beta),
                "order_var_fraction_beta": float(bt.order_var_fraction_beta),
                "anchor_pony": cfg.anchor_pony,
                "symmetrize": cfg.symmetrize,
                "beta": [float(b) for b in bt.beta[order]],
            },
        )

    def save_artifact(self, output_dir: str | Path) -> None:
        from llm4rec.io.artifacts import write_json

        write_json(Path(output_dir) / "parc_config.json", {**self.config.to_dict(), "lambda": self.lam})
