# Phase 10 Gates

Use this reference for milestone planning, reviewer-style checks, and decisions about whether experiments are ready for writing.

## Active Direction

TGL-Rec is in Phase 10: observation to original framework to full recommender system.

The research path is:

1. Measure whether LLM recommenders rely on semantic similarity and popularity more than temporal need transitions.
2. Build the original TGL-Rec framework around temporal directed item graph evidence, graph-to-language translation, need-aware gating, and candidate-grounded local Qwen3-8B reranking.
3. Compare against strong, faithful baselines under one frozen protocol.
4. Scale to four large same-candidate domains: beauty, books, electronics, and movies.
5. Pass reviewer and reproducibility gates before paper writing.

## Milestone Map

- M0/M0.5: large-scale observation matrix on the frozen same-candidate protocol, including base Qwen3-8B, fixed-label-mask controls, perturbations, time-tag ablations, and faithful senior-reference probes when available.
- M1: original framework modules, including train-only temporal graph construction, time-windowed/decayed transition evidence, semantic-vs-transition separation, need gating, evidence translation, and ablation switches.
- M2/M2.5: full recommender comparison with non-personalized, traditional, sequential, text retrieval, LLM, and faithful official reference baselines.
- M3: four-domain large protocol import, validation, and evaluation.
- M4: top-conference reviewer gate for novelty, fairness, rigor, leakage, reproducibility, and statistical evidence.

## Reviewer Pressure Test

Before claiming readiness, check that:

- novelty is not just prompt wording, timestamps, or a recombination of senior-reference papers;
- the method has distinct mechanisms with ablations and auditable provenance;
- baselines preserve official algorithms where possible and are labeled honestly when blocked or reproduced;
- all methods share candidate rows, splits, metric code, prediction schema, and event IDs;
- no valid/test information leaks into graph evidence, training data, retrieval, prompts, or tuning;
- paired statistics, diagnostics, and table exports come from saved artifacts;
- failure cases and limitations are documented from real diagnostics.

## Experiment Ending Gate

Do not say the experiment phase is basically complete until all of these are true:

- the large observation matrix has run on the frozen four-domain protocol or a documented replacement;
- TGL-Rec framework configs, ablations, diagnostics, paired statistics, and exported tables exist;
- at least four faithful official/senior baselines are implemented, or missing ones are blocked with reviewer-acceptable reasons and replacements;
- leakage, reproducibility, candidate-alignment, and significance checks pass;
- a top-conference-style reviewer pass finds no blocking P0/P1 issue.
