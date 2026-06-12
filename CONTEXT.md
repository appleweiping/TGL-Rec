# Project Context — TGL-Rec

## Current State (as of 2026-06-12)

| Metric | Value |
|--------|-------|
| GitHub | https://github.com/appleweiping/TGL-Rec |
| Branch | main (CC-PACE mainline; phase9e branches are historical) |
| Stage | Phase 10 — CC-PACE beauty-first run (data seams wired, ready for server) |
| Official baselines | 8 methods × 8 domains, frozen (`data/pony_official_baselines/`) |
| Setting | 8 domains, 101-candidate same-candidate, Qwen3-8B; beauty bar = promax NDCG@10 0.1506 |
| Method | CC-PACE (Collaborative-Conditioned Panel-Anomaly Calibrated Exchangeability reranker) |
| LLM | Qwen3-8B (frozen judge; LoRA via Plackett-Luce after zero-shot GO) |
| Python | >=3.10 |
| License | MIT |

## Research Question
Do LLM rerankers score candidates by user-conditional relevance, or collapse to
popularity/semantic similarity? (The old temporal-graph premise is abandoned; see CLAUDE.md.)

## Key Decisions
- Same-candidate protocol frozen: all methods rank the same 101 candidates per user
- CF is a frozen conditioning σ-field (SASRec artifacts), never a score head
- Evidence levels enforced: no paper claims without controlled experiments + significance
- 2026-06-12: CF artifacts + profile slots data prep wired locally
  (`scripts/build_cc_pace_cf_artifacts.py`, `scripts/build_cc_pace_profiles.py`,
  `methods/cc_pace/text_facets.py`; driver takes `--cf-artifacts/--profiles`); 18 CPU tests green

## What's Next
- [ ] Server: pull main, build CF/profile artifacts, run beauty zero-shot kill test
      (`scripts/run_cc_pace_beauty.sh 0`; GO = NDCG@10 ≥ 0.13 AND full > text_only)
- [ ] If GO: LoRA training (Plackett-Luce + dCor), beauty formal eval vs 0.1506
- [ ] If SOTA: roll out to 7 domains; else re-run 3-seat ARIS redesign
- [ ] Then: observation / ablation / hyperparameter experiments + overview figure + paper
