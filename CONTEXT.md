# Project Context — TGL-Rec

## Current State (as of 2026-06-13)

| Metric | Value |
|--------|-------|
| GitHub | https://github.com/appleweiping/TGL-Rec |
| Branch | feat/cc-pace-data-seams (active server-sync branch; main promotion pending) |
| Stage | Phase 10 — CC-PACE beauty-first zero-shot gate (text_only done; full resume blocked by HF judge OOM fix/retest) |
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
- 2026-06-13: Server `text_only` finished at NDCG@10 = 0.1108; `full` resumed to 65/973 users then
  hit Qwen3 KV-cache OOM before writing `full.json`/`go_verdict.json`. HF judge now uses a read-only
  expanded-cache wrapper plus adaptive halving/per-panel CUDA cleanup so suffix scoring does not
  retain prompt-sized cache copies; server conda equivalence retest and full resume are the next gate.

## What's Next
- [ ] Server: sync `feat/cc-pace-data-seams`, run HF judge equivalence tests in `tglrec-lora`, then
      resume only `full` from `outputs/cc_pace_beauty/full.json.per_user.jsonl` and write
      `outputs/cc_pace_beauty/go_verdict.json`
- [ ] If GO: LoRA training (Plackett-Luce + dCor), beauty formal eval vs 0.1506
- [ ] If SOTA: roll out to 7 domains; else re-run 3-seat ARIS redesign
- [ ] Then: observation / ablation / hyperparameter experiments + overview figure + paper
