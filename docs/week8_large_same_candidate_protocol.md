# Week8 Large Same-Candidate Protocol

## Purpose

This note records the intended large-scale evaluation protocol that should
replace the current toy-ish debugging data for paper-scale claims.

The data is being produced by the adjacent server project:

```text
~/projects/pony-rec-rescue-shadow-v6
```

The TGL-Rec project must treat this as a frozen external task package. Do not
resample users, positives, negatives, or candidate sets when importing it.

## Current External Task Location

Expected task directories:

```text
~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/{domain}_large10000_100neg_{valid,test}_same_candidate/
```

Target domains:

- `books`
- `electronics`
- `movies`

The user also has a complete `beauty` domain on the server. It should be
integrated as a separate frozen domain/protocol input once its exact task layout
is confirmed.

Check available files on the server with:

```bash
find ~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks \
  -path "*large10000_100neg*" -type f | sort
```

External project summaries are expected under:

```text
~/projects/pony-rec-rescue-shadow-v6/outputs/summary/
```

## Protocol Characteristics

Known construction details from the adjacent project:

- target domains: `books`, `electronics`, `movies`;
- maximum users per domain: 10,000;
- each ranking event has 1 positive and 100 negatives;
- same-candidate setting: all methods score the same candidate set for each
  event;
- negative sampling: popularity;
- test history mode: `train_plus_valid`;
- seed: `20260506`;
- shuffle seed: `42`.

This is a protocol data package, not merely a result table. It is suitable for
paired comparison, oracle/rank-fusion analysis, and statistical testing because
events and candidates are aligned across methods.

## Expected Files Per Task Directory

Each task directory is expected to contain:

- `ranking_valid.jsonl` or `ranking_test.jsonl`: event-level candidate ranking
  tasks;
- `candidate_items.csv`: candidate set per user/event;
- `train_interactions.csv`: train interactions;
- `item_metadata.csv`: item metadata;
- `selected_users.csv`: sampled users;
- `recbole/{dataset}.inter`: RecBole-format training data;
- `metadata.json`: protocol metadata.

## Required Import Rules

When adding this protocol to TGL-Rec:

- read the external task directories as immutable inputs;
- preserve event and candidate alignment exactly;
- do not regenerate negatives;
- do not resample users;
- do not alter split membership;
- preserve `event_id` or `source_event_id` whenever present;
- preserve `user_id`, `item_id`, `split`, and domain;
- save the imported protocol under a new frozen version, for example
  `protocol_week8_large10000_same_candidate`, rather than overwriting
  `protocol_v1`.

## Framework Work Needed

The next implementation stage should add an importer/adapter that converts the
external task package into the TGL-Rec interfaces:

1. Load `ranking_valid.jsonl` and `ranking_test.jsonl` as event-level
   `UserExample` records.
2. Load `candidate_items.csv` as the authoritative candidate set for each event.
3. Load `train_interactions.csv` for user histories and SFT training data.
4. Load `item_metadata.csv` for prompt text and text-based baselines.
5. Preserve external event IDs in predictions for paired comparison.
6. Emit frozen artifacts under `outputs/artifacts/<new_protocol>/<domain>/`.
7. Rebuild LoRA SFT datasets from training-only data.
8. Evaluate every baseline with the shared evaluator and same candidate sets.

Importer entrypoint:

```bash
python scripts/import_week8_same_candidate.py \
  --task-dir ~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/books_large10000_100neg_test_same_candidate \
  --task-dir ~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/electronics_large10000_100neg_test_same_candidate \
  --task-dir ~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/movies_large10000_100neg_test_same_candidate \
  --protocol-version protocol_week8_large10000_same_candidate
```

The prediction JSONL schema should be extended, without breaking existing
fields, to carry:

```json
{
  "event_id": "...",
  "source_event_id": "...",
  "split": "valid|test"
}
```

inside either top-level fields or `metadata`, consistently across methods.

## Relationship To Current Results

Current `protocol_v1` data (`movielens_full` and
`amazon_multidomain_filtered_iterative_k3`) should remain useful for framework
debugging, smoke tests, and diagnostics, but not for final paper-scale claims.

The large same-candidate protocol should become the main evidence base before
making claims about our observation, reference-paper baselines, or cross-domain
behavior.
