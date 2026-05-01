# Lightweight Temporal Graph Encoder

Phase 6 adds `TemporalGraphEncoder`, a lightweight trainable dynamic graph encoder option for
smoke and sampled validation. It is inspired by dynamic graph recommendation event processing but
is not a full TGN implementation.

## Architecture

- User memory embeddings.
- Item memory embeddings.
- Log timestamp projection.
- GRUCell-based user and item memory updates.
- Dot-product user-item scoring with timestamp context.
- `update(event)` for inductive event updates.

## Safety

Training events must be train-only for reportable runs. Prediction-time use must not include
future events after the prediction timestamp. Phase 6 configs remain `reportable=false`.

## Smoke Command

```bash
python scripts/train_temporal_graph.py --config configs/experiments/phase6_temporal_graph_smoke.yaml
```

If PyTorch is unavailable, the command writes a clear skipped artifact. No paper-scale results or
empirical superiority claims are made.
