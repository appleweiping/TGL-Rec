# Baseline And Evaluation Contracts

Use this reference when touching baselines, prediction outputs, score imports, metrics, protocol artifacts, or paper tables.

## Frozen Same-Candidate Protocol

The active external task root on the shared server is:

```text
~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/
```

Task families:

```text
beauty_supplementary_smallerN_100neg
books_large10000_100neg
electronics_large10000_100neg
movies_large10000_100neg
```

Treat `ranking_valid/test.jsonl` and `candidate_items.csv` as immutable. Do not resample users, resample negatives, alter candidate membership, alter candidate order, or tune on test.

Every imported method score file must use:

```text
source_event_id,user_id,item_id,score
```

Import same-candidate baseline/model scores through:

```text
main_import_same_candidate_baseline_scores.py
```

Use an explicit protocol version such as `protocol_week8_large10000_same_candidate`.

## Reportable Baseline Rules

Reportable baselines must use the same split, candidates, event IDs, metric implementation, prediction schema, and paired comparison rows as TGL-Rec.

For senior-reference LLM baselines:

- prefer official code;
- preserve official losses, heads, adapters, ID tokens, distillation objectives, evidence construction, and scoring logic where they define the method;
- use Qwen3-8B/project LoRA or QLoRA policy only where faithful;
- log official/default or paper-recommended hyperparameters;
- keep TGL-Rec validation tuning separate and logged.

Do not use `reference_*_sft` scaffold variants as main-table baselines. They are scaffolds or controls unless explicitly promoted with provenance and review.

## Required Prediction Shape

Shared prediction JSONL rows should preserve:

```json
{
  "user_id": "u1",
  "target_item": "i9",
  "candidate_items": ["i1", "i2", "i9"],
  "predicted_items": ["i9", "i2", "i1"],
  "scores": [0.9, 0.5, 0.1],
  "method": "method_name",
  "domain": "movies",
  "raw_output": null,
  "metadata": {}
}
```

When the same-candidate importer is used, also preserve `event_id` or `source_event_id` so paired metrics remain possible.

## Output Artifacts

Each meaningful run should save resolved config, environment, git info, logs, predictions, metrics JSON/CSV, cost/latency details, checkpoints when applicable, and artifacts under an explicit run directory.

Never type paper tables by hand from memory. Export from saved metrics and prediction artifacts.
