# Reference LoRA Baseline Plan

## Goal

The `references/` papers should inform baseline methods, but their results must
not be copied as external numbers. Each reference-style baseline should be
reimplemented as a variant inside this project's framework and trained/evaluated
under the same local small-model setting:

- base model: Qwen3-8B;
- adaptation: LoRA/QLoRA;
- data split: project `protocol_v1` or a later frozen protocol;
- candidate construction: shared project candidate protocol;
- output schema: shared `predictions.jsonl`;
- evaluator: shared ranking, validity, hallucination, and diagnostic metrics.

This makes the comparison a framework-controlled experiment rather than a
literature-number comparison.

## Current Reference Inputs

Local reference material is stored under:

- `references/NH/`
- `references/NR/`
- `references/recprefer.zip`

The files are local research references and are not committed. Lightweight notes
derived from them may be committed only as manually written summaries.

## Baseline Families To Adapt

Initial reference-style LoRA variants are registered in
`src/llm4rec/trainers/sft_variants.py`:

| Variant | Role | Intended Reference Family |
|---|---|---|
| `history_only_sft` | control | plain sequential ID-history LoRA |
| `temporal_evidence_sft` | ours/observation | temporal transition and contrastive evidence |
| `reference_preference_sft` | reference baseline | preference alignment, controllability, instruction preference |
| `reference_semantic_sft` | reference baseline | semantic/text/multimodal matching |
| `reference_long_tail_sft` | reference baseline | long-tail and popularity-bias mitigation |

These are not paper claims yet. They are framework slots. A variant becomes a
reportable baseline only after a specific reference paper is mapped to it with:

- paper identity and citation;
- which signal is adapted;
- prompt/evidence fields;
- label construction policy;
- expected observation axes;
- smoke run;
- server LoRA run;
- saved metrics and diagnostics.

## Observation-Aware Evaluation

The main question is not only whether our variant beats baselines. We must also
test whether the observed phenomenon appears in other LoRA baselines.

For every Qwen3-8B LoRA variant, report:

- ranking metrics: Recall, NDCG, HitRate, MRR;
- output quality: parse success, validity, hallucination, candidate adherence;
- diagnostic behavior: prompt continuation rate, JSON-like rate, bare-ID rate;
- observation axes when available: temporal transition sensitivity, semantic
  similarity reliance, long-tail behavior, popularity bias, and candidate
  grounding.

If a phenomenon appears in reference baselines too, the paper should describe it
as a broader LLM4Rec behavior rather than claiming it is unique to our method.
If it appears only in our variant, then the ablation needs to show which evidence
component caused it.

## Dataset Position

The current `movielens_full` and `amazon_multidomain_filtered_iterative_k3`
artifacts are useful for framework debugging and diagnostic runs, but they
should be treated as preliminary. The later conference-grade dataset generated
by the adjacent server project should be integrated as a new frozen protocol
version rather than patched into existing results.

Expected future integration steps:

1. Convert the new dataset into the project interaction/item schema.
2. Run readiness checks and leakage audits.
3. Freeze splits and candidates under a new protocol version.
4. Rebuild all LoRA SFT variants from train split only.
5. Train every baseline variant with the same Qwen3-8B LoRA settings.
6. Evaluate all variants with the same evaluator and diagnostic scripts.

## Near-Term Plan

1. Finish reference paper indexing into lightweight notes.
2. Map each selected paper to one of the registered variant families or add a
   new variant only when the paper requires a genuinely different signal.
3. Retrain the fixed-label-mask LoRA variants before trusting any LoRA result.
4. Run `limit=20` diagnostics first, then `limit=200` only after output behavior
   is stable.
5. Defer paper-scale claims until the stronger dataset is available and frozen.
