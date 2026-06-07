# Official Baseline Evidence

This directory holds the frozen evidence for the **eight official LLM4Rec baselines** that
our method is compared against, evaluated under our standard ranking protocol across the
**eight benchmark domains**. It is the single source of truth for every baseline number that
appears in the paper's main comparison table.

Heavy artifacts (per-event score matrices, model checkpoints, raw prediction dumps) are kept
on the compute node by design; only the lightweight, paper-relevant evidence is committed here
so the comparison is fully reproducible from version control without storing multi-GB files.

## Evaluation protocol

All baselines and our method are evaluated identically:

- **Task**: candidate ranking (rerank a fixed candidate set per user event).
- **Candidates**: same-candidate protocol, **101 candidates per user** (1 positive + 100 negatives).
- **Backbone**: Qwen3-8B for all LLM-based baselines.
- **Users**: 10,000 test users per domain (Beauty is a supplementary smaller-N set, 973 users).
- **Comparison variant**: `official_code_qwen3base_default_hparams_declared_adaptation` —
  each baseline is run from its **official code at a pinned commit** with default hyper-parameters
  and a declared, audited adaptation to our candidate schema.
- **Score coverage**: 1.0 (every candidate scored) for all completed rows.

## Metrics

Reported per (domain, baseline): **HR@5/10/20**, **NDCG@5/10/20**, **MRR**. Ranking-quality
metrics are the primary axis; exposure/coverage diagnostics are retained per baseline in
`external_score_coverage.csv` and `ranking_metrics.csv`.

## The eight official baselines

| key | method | official repo |
|-----|--------|---------------|
| `elmrec`  | ELMRec (graph-enhanced)        | graph-style LLM4Rec |
| `irllrec` | IRLLRec (intent)               | intent-aware LLM4Rec |
| `llm2rec` | LLM2Rec (SASRec-style)         | sequential LLM4Rec |
| `llmemb`  | LLMEmb (embedding alignment)   | https://github.com/Applied-Machine-Learning-Lab/LLMEmb |
| `llmesr`  | LLM-ESR (SASRec-enhanced)      | LLM-enhanced sequential rec |
| `proex`   | ProEx (profile/explanation)    | profile-based LLM4Rec |
| `promax`  | ProMax (profile)               | profile-based LLM4Rec |
| `rlmrec`  | RLMRec (graph contrastive)     | representation-learning LLM4Rec |

Exact official repository URLs, pinned commits, entrypoints and audited adaptations are recorded
per baseline in each `fairness_provenance.json`.

> The 8th baseline slot is **`llmemb`**. Earlier drafts of this project used `setrec` here; it
> has been replaced by `llmemb` so the baseline set matches the protocol used across all eight
> domains (`setrec` only had a single-domain run and is not part of the main comparison).

## The eight domains

| group | domains | users |
|-------|---------|-------|
| primary    | sports, toys, home, tools             | 10,000 each |
| additional | books, electronics, movies            | 10,000 each |
| additional | beauty                                | 973 (supplementary smaller-N) |

## Files

```
baseline_comparison_8domains.csv   # master table: 64 rows (8 domains x 8 baselines), one row per pair
IMPORT_MANIFEST.json               # integrity manifest: every committed file with size + sha256
domains/<domain>/<baseline>/
    same_candidate_external_baseline_summary.csv  # full metric row (HR/NDCG/MRR + audit fields)
    ranking_metrics.csv                           # ranking metrics only
    external_score_coverage.csv                   # candidate score-coverage audit
    fairness_provenance.json                      # official repo, pinned commit, adaptation (where available)
    <baseline>_official_run_summary.json          # run metadata
    <baseline>_official_score_audit.json          # score-file validation
```

The master table is regenerated from the per-pair summaries by
[`scripts/build_official_baseline_table.py`](../../scripts/build_official_baseline_table.py),
which asserts exactly 64 baseline rows and that no internal-method rows are present.

## Provenance completeness

All 64 (domain, baseline) pairs have the metric summary and coverage audit. For the four primary
domains every pair additionally carries `fairness_provenance.json` and the run/score-audit JSONs.
For the additional domains some pairs currently retain the metric summary and coverage only
(directories with 3 files vs 6); the per-pair file inventory is enumerated in `IMPORT_MANIFEST.json`.

## Status

This evidence makes the **baseline comparison** available and frozen. It does **not** by itself
establish where our method stands relative to these baselines — that depends on our method's own
runs under the same protocol, which are tracked separately. Treat the numbers here as the fixed
reference our method is measured against.

