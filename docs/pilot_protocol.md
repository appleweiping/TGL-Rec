# Phase 7 Pilot Protocol

The pilot matrix is a NON_REPORTABLE systems validation run. It checks that methods can share one
split, one candidate artifact, one prediction schema, and one evaluator.

Pilot defaults:

- MovieLens-style sampled pilot fixture.
- `max_users: 200`
- `max_items: 1000`
- `max_interactions: 10000`
- `candidate_size: 100`
- seed `[0]`
- `allow_api_calls: false`
- `enable_lora_training: false`

The local pilot fixture is deterministic and saved under each run's `artifacts/processed_dataset`.
It is not a substitute for MovieLens full or Amazon full paper-scale runs.
