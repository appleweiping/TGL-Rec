# Reference Baseline Fidelity Rules

## Non-Negotiable Baseline Rule

The senior-recommended baselines must be original reference-method adaptations,
not toy prompt variants.

The governing principle is:

```text
Training/scoring algorithm: preserve the baseline's own logic as much as possible.
Experimental protocol: unify data, candidates, splits, metrics, prediction schema,
Qwen3-8B backbone, and the project LoRA/QLoRA regime for the main LLM baseline table.
```

Do not make a baseline artificially weak by flattening its algorithm into a
generic prompt. Do not make the comparison uncontrolled by letting each baseline
choose its own data, negatives, candidate set, split, metrics, backbone, or
fine-tuning regime.

## Main Paper Fairness Setting

The default reportable setting follows the senior-recommended academic practice:

```text
official source code + our frozen data/protocol + Qwen3-8B backbone +
project LoRA/QLoRA regime + official/default baseline hyperparameters +
validation-tuned TGL-Rec with all search ranges logged
```

This is the primary comparison table unless the paper explicitly opens a
separate regime. We do not claim equal hyperparameter-search budget. Baselines
use official/default or paper-recommended hyperparameters, except for documented
dataset-adaptation values that are required to run on the frozen protocol. Our
method may be tuned on validation data, but every searched range, selected
setting, seed, and failed setting must be logged.

If a baseline's core contribution is inseparable from LoRA/adapter design, the
official design should be adapted into the project LoRA/QLoRA regime as closely
as possible and the deviation recorded. If a full-finetune table is added later,
all LLM methods in that table must use the same full-finetune regime; do not
compare our full-finetuned method against LoRA-only baselines in the main table.

Official implementation rule:

- Use the official code/project whenever it exists.
- TGL-Rec should wrap official training/scoring code with adapters for the
  shared protocol rather than reimplementing the algorithm from scratch.
- A paper without identified official code cannot be used as a main
  senior-recommended baseline unless the user explicitly approves a
  non-official reproduction and it is labeled as such.
- A local rewrite is allowed only for glue code: data conversion, candidate
  alignment, Qwen3-8B base-model adaptation, prediction-schema export, and metric
  evaluation.

Fairness means every reportable reference baseline uses the same protocol base
model and evaluator:

- backbone: Qwen3-8B for LLM-based reportable baselines;
- fine-tuning regime: project LoRA/QLoRA for the main LLM table;
- baseline hyperparameters: official/default or paper-recommended values;
- adaptation/training/head/scorer: preserve each official baseline algorithm
  inside that regime where possible;
- same train/valid/test split;
- same candidate sets;
- same prediction schema;
- same evaluator and diagnostics.

But matching the base model is not enough. A baseline is reportable only if its
method signal comes from a concrete reference paper or project.

## What Should Be Preserved

For each original reference baseline, preserve as much of its own method as
possible:

- its training objective;
- its input/evidence construction;
- its preference, semantic, collaborative, long-tail, distillation, or control
  signal;
- its scoring or reranking logic;
- its ablations when feasible.

Only adapt the parts needed for a fair shared protocol:

- replace the original LLM backbone with Qwen3-8B for the main Qwen3-LoRA table;
- adapt LoRA/adapter/head/scorer training to the project LoRA/QLoRA regime while
  preserving the method's official algorithmic signal;
- use official/default or paper-recommended baseline hyperparameters;
- replace original data splits/candidates with our frozen protocol;
- emit the shared prediction schema;
- evaluate with the shared evaluator.

If a paper's baseline cannot be faithfully adapted to the shared Qwen3-8B
base-model protocol without destroying its method, record that limitation
instead of forcing a toy version.

## Current Status

The following variants are currently control/method variants and are reportable
after their actual runs complete:

- `history_only_sft`: control group.
- `temporal_evidence_sft`: our temporal/contrastive evidence variant.

The following variants are currently only candidate scaffolds:

- `reference_preference_sft`
- `reference_semantic_sft`
- `reference_long_tail_sft`
- `reference_collaborative_sft`

They must not be described as senior-recommended original baselines yet. They
exist only as implementation containers for future faithful adaptations.

## Promotion Criteria

A candidate scaffold becomes a reportable reference baseline only after all of
the following are true:

1. It is mapped to a specific reference paper or project from `references/`.
2. The paper identity, title, venue/year if known, and local file path are
   recorded in a committed note.
3. The official code path is identified, cloned or referenced, and its license
   or usage constraints are recorded.
4. The adapted method signal is described concretely, not generically.
5. The baseline's own training, adapter/head/loss, and scoring logic is
   preserved as much as the shared data/candidate/evaluator protocol allows.
6. The implementation uses Qwen3-8B and the project LoRA/QLoRA regime for the
   main LLM table, or documents why it must be excluded or reported separately.
7. A smoke build verifies the data path and prompt fields.
8. A server training run produces an adapter or baseline checkpoint.
9. The adapter/checkpoint is evaluated with the shared evaluator.
10. Metrics and diagnostics are saved.

Until then, these variants are framework scaffolds only.

## Candidate Reference Methods To Inspect

The local reference set includes identifiable candidates that may become real
baselines after method extraction:

- `references/NH/11465_SLMRec_Distilling_Large_.pdf`
  - visible filename: SLMRec / distilling large language models.
  - likely direction: small-model distillation for recommendation.
- `references/NH/Aligning Large Language Models for Controllable Recommendations.pdf`
  - direction: controllable recommendation / preference or control alignment.
- `references/NH/NeurIPS-2024-llm-esr-large-language-models-enhancement-for-long-tailed-sequential-recommendation-Paper-Conference.pdf`
  - direction: LLM-ESR / long-tailed sequential recommendation.
- `references/NR/3589334.3645347.pdf`
  - metadata title: Collaborative Large Language Model for Recommender Systems.
  - direction: collaborative LLM recommendation.
- `references/NR/3589334.3645458.pdf`
  - metadata title: Representation Learning with Large Language Models for
    Recommendation.
- `references/NR/3637528.3671884.pdf`
  - metadata title: Bridging Items and Language: A Transition Paradigm for
    Large Language Model-Based Recommendation.
- `references/NR/3726302.3730055.pdf`
  - metadata title: Review-driven Personalized Preference Reasoning with Large
    Language Models for Recommendation.

This list is only an inspection queue. It is not yet a completed baseline map.

The selected adaptation map is maintained in
`docs/reference_method_adaptation_map.md`. That map still does not make a
baseline reportable; it only fixes which original methods should be implemented
first.
