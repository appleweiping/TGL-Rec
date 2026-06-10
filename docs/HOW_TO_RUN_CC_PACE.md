# CC-PACE — How To Run (handoff guide for any agent)

This is the single entry point for running CC-PACE experiments and writing the paper.
Everything is implemented; an agent should be able to go straight to experiments. Read
`docs/method_v2_decision_CC-PACE.md` for the why, this file for the how.

## TL;DR pipeline
1. Baselines are DONE and frozen: `data/pony_official_baselines/` (8 domains × 8 official baselines,
   64-row master table). Beauty SOTA bar to beat = **promax NDCG@10 = 0.1506**.
2. Method = **CC-PACE** (`src/llm4rec/methods/cc_pace/`, ranker `llm4rec.rankers.CCPaceRanker`).
3. Run **beauty first** (zero-shot probe → ablations → LoRA if GO) → if SOTA, roll to the other 7
   domains. If not SOTA → re-run the 3-seat ARIS discussion, redesign, re-run beauty until SOTA
   (see `multi-agent-discussion-rule` in global memory).
4. After the performance table: do the 3 required experiments in `docs/paper_followup_experiments.md`
   (observation / ablation / hyperparameter) + an overview figure → submit.

## Discipline (non-negotiable; see global memory feedback_local_server_alignment)
- Experiments run on the SERVER; commit/push only from LOCAL.
- After each server run, scp lightweight evidence (metrics JSON/CSV, provenance) back to local;
  heavy artifacts (scores, checkpoints) stay server-side.
- Don't stop unless: server disk full / API failure / unfixable review reject / user says pause.
- Every chunk of work → update agentmemory + local docs + README.

## Code map (`src/llm4rec/methods/cc_pace/`)
| file | role |
|------|------|
| `config.py` | `CCPaceConfig` — every hyperparameter + `use_*` ablation switches |
| `schema.py` | unified candidate schema, label randomization (exchangeability), profile slots |
| `judge.py` | forced-choice listwise evidence extraction (`EvidenceModel` protocol) |
| `hf_judge.py` | concrete frozen Qwen3-8B model (server/GPU only; lazy torch import) |
| `cf_conditioning.py` | frozen CF as conditioning σ-field: evidence tokens + nuisance coords |
| `residualizer.py` | symmetric-LOO isotonic (additive) + rich (ceiling-test) residualizers |
| `conformal.py` | per-candidate conformal p, dual-null merge, James-Stein shrinkage |
| `statistic.py` | orchestrator: evidence → residual → shrink → ranked T_u (+ p_pop) |
| `llm4rec.rankers.cc_pace.CCPaceRanker` | BaseRanker entry point for the eval harness |
| `llm4rec.trainers.cc_pace_trainer` | Plackett-Luce loss + dCor penalty + training plan |

## Running the beauty experiment

### Local CPU plumbing/CI (mock judge — NOT for reporting)
```bash
python scripts/cc_pace_beauty.py --task <ranking_test.jsonl> --out outputs/cc_pace_beauty/full.json --mock --variant full
```

### Server zero-shot kill test (frozen Qwen3-8B, no training)
```bash
# only launches if >=17GB GPU free (never preempts another job)
bash scripts/run_cc_pace_beauty.sh 0        # 0 = all 973 users; small int = sanity subset
```
This runs `full` + the ablations `text_only`, `no_residualizer`, `no_shrinkage`, `rich_residualizer`
and writes one JSON each under `outputs/cc_pace_beauty/`.

### Go / kill (vs SOTA bar 0.1506) — from docs/method_v2_decision_CC-PACE.md
- **Zero-shot probe** (no training): if the residualized T carries no signal vs popularity → revisit
  before spending GPU on LoRA.
- **GO to LoRA:** zero-shot NDCG@10 ≥ 0.13 AND the CF-token ablation (`full` vs `text_only`) shows a
  positive gap.
- **STRONG GO / reportable:** post-LoRA NDCG@10 ≥ 0.1506 with paired-bootstrap p<0.05 over users, AND
  panel-corruption drops ≥30%, AND `text_only` falls below the CF baseline (proves the mechanism).
- **KILL / reframe:** only wins on popularity-heavy slice; or conformal coverage breaks; or T→0 under
  `rich_residualizer` and only late-fusion works → reposition as "panel-exchangeable calibration of a
  hybrid score" (honest fallback).

## What still needs wiring before a REAL run (the only open implementation work)
The method, ranker, scripts, tests, and ablation switches are all implemented and CPU-tested. Two
pieces need real data plumbing on the server (deliberately left as thin, documented seams):

1. **CF provider artifacts** (`cf_conditioning.PrecomputedCFProvider`): produce, offline, the frozen
   CF neighbours + scores + clusters per (user, candidate) from an existing collaborative model (the
   repo already has a SASRec ranker/trainer). Populate the `scores/neighbors/clusters` dicts (or a
   small loader) and pass the provider into `CCPaceRanker(..., cf_provider=...)`. Until then the ranker
   uses `NullCFProvider` = text-only PACE (a valid ablation, not the headline method).
2. **Profile slots** (`CCPaceRanker.set_profiles`): compress each user's TRAIN history into the profile
   slots (top categories, liked brands, concerns, routine step, ingredient prefs, price band). A simple
   history-aggregation pass; mandatory for beauty (promax is profile-strong).

Both are pure data-prep from artifacts the project already produces. The `--mock` driver path proves
the full statistic→metric pipeline end-to-end without them.

## LoRA training (after a GO)
`llm4rec.trainers.cc_pace_trainer.build_training_plan(cfg)` emits the declarative plan; the loss is
`plackett_luce_top1_loss` (+ optional `distance_correlation` popularity penalty). Reuse the repo's
existing HF LoRA loop (`llm4rec.trainers.lora` / `lora_sft`) to apply the adapter, then evaluate with
the SAME driver via `--adapter <path>` (no `--mock`). Train panels come from TRAIN interactions only
(pseudo-held-out positive + popularity-matched & intent-matched negatives); split-conformal = train on
fold A, calibrate on disjoint fold B. The test panel is never seen in training.

## Rolling to the other 7 domains
Once beauty is SOTA, the same ranker + driver run unchanged on sports/toys/home/tools/books/
electronics/movies — point `--task` at each domain's `ranking_test.jsonl` and compare against that
domain's rows in `data/pony_official_baselines/baseline_comparison_8domains.csv`. Run domains serially
(server disk). Each domain's numbers + the ablations populate the paper's main table.

## Tests
`python -m pytest tests/unit/test_cc_pace_core.py tests/unit/test_cc_pace_ranker.py -q` (CPU, fast):
conformal validity, residualizer popularity-removal, shrinkage, schema exchangeability, end-to-end
ranker (mock judge), PL loss/grad.
