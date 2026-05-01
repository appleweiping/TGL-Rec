# Experiment Gates

This project should not jump straight to LLM APIs or LoRA. The setup path is:

1. **Local CPU, no API**: deterministic data, leakage checks, sanity baselines, BPR-MF,
   TDIG candidate recall, semantic-vs-transition stress cases, and RecBole general-CF export.
2. **GPU/server, no API**: reproduce LightGCN, GRU4Rec, SASRec, BERT4Rec, and TiSASRec or a
   verified time-aware alternative under the project splits.
3. **Small reranker, no API**: train the original TDIG/need-aware gate using candidate, graph,
   time, semantic, and popularity features.
4. **Local LLM or LoRA**: only after strong non-LLM baselines and TDIG reranker results exist.
   Use it for LLM-SRec-style reproduction or controlled graph-to-language evidence ablations.
5. **Hosted/proprietary API**: optional and late. Use only after user approval of cost/data policy,
   with prompt hashes, caching, request logs, latency, and a non-API comparison.

The machine-readable gate is `configs/stage_gates.yaml`.

## Basic Setup Is Not Done Until

- MovieLens plus at least one need-transition-heavy domain is preprocessable with manifests.
- Local popularity, item-kNN, BPR-MF, and BPR-MF sweep commands are documented and tested.
- RecBole general-CF export is ready for BPR/LightGCN-style baselines.
- Sequential baseline export or a runner adapter is ready for SASRec/BERT4Rec/TiSASRec.
- TDIG graph construction, candidate recall, and semantic-vs-transition stress cases exist.
- All experiment runners write configs, command, seed, git commit, metrics, segment metrics, and
  checksums under `runs/` or `artifacts/`.
- `BASELINES.md`, `EXPERIMENTS.md`, and `TASKS.md` match the implemented commands.

## Originality Guardrails

The original contribution must remain the diagnosis plus temporal graph-to-language retrieval and
need-aware gating. API calls, LoRA, and external baselines are measurement tools or comparators, not
the core novelty. Do not claim SOTA unless the strongest baselines are fairly rerun under the same
split, candidate/full-ranking protocol, and hyperparameter-search budget.
