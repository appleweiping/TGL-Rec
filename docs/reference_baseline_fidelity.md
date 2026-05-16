# Pony Official Baseline Fidelity Rules

## Active Policy

TGL-Rec no longer treats the old senior-reference LoRA scaffold queue as the
main baseline plan. The active baseline source is the Pony/Uncertainty official
same-candidate suite recorded in:

```text
configs/baselines/pony_official_external.yaml
```

This is a reuse and migration policy, not a license to copy result tables by
hand. Every reportable row still needs score/provenance artifacts, exact
candidate-key checks, shared metrics, and paired event IDs.

## Main Baseline Suite

Completed main-table candidates:

- `llm2rec`
- `llmesr`
- `llmemb`
- `rlmrec`
- `irllrec`
- `elmrec`
- `proex`

Pending:

- `promax`: final 2026 official baseline, planned but excluded from completed
  main tables until all declared domains pass exact-score gates.

Blocked/replaced:

- `setrec`: blocked by upstream large-domain failure and replaced by `elmrec`,
  `proex`, and `promax`.

## Fidelity Contract

The comparison policy is:

```text
Pony official-code or official-code-level implementation
+ shared same-candidate train/valid/test protocol
+ Qwen3-8B declared-adaptation policy where LLM/text representations are used
+ official/default or recommended baseline hyperparameters
+ exact source_event_id,user_id,item_id,score export
+ TGL-Rec shared evaluator and paired statistics
```

Do not import full-catalog metrics from external repositories into the
same-candidate table. Scores may be method-native and uncalibrated, but they must
rank candidates within each event and pass exact key coverage.

## Historical Appendix

The old `reference_preference_sft`, `reference_semantic_sft`,
`reference_long_tail_sft`, and `reference_collaborative_sft` configs are
historical/non-reportable scaffolds. They may remain for provenance and tests,
but they are not active senior-reference baselines and must not enter main
tables.

Older selected methods such as SLMRec, CLLM4Rec, TransRec, and review-preference
reasoning can still inform related work or future supplements, but they no
longer drive the Phase 10 baseline execution plan unless the user explicitly
changes strategy.

## Artifact Policy

Do not copy large Pony evidence archives, checkpoints, or embeddings into git.
TGL-Rec should record manifest paths, hashes or sizes when available, summary
tables, import status, and score-gate outcomes. The second migration stage may
bring over Pony's runner/importer design, but large artifacts remain external.
