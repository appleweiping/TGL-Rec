# Sequential Baselines

Phase 6 adds a real SASRec-style sequential recommender option. It is implemented in
`src/llm4rec/models/sasrec.py`, trained through `scripts/train_sasrec.py`, and evaluated with the
shared prediction schema and shared evaluator.

## SASRec Status

- Real PyTorch model when the optional `torch` dependency is installed.
- Includes item embeddings, positional embeddings, causal self-attention, feed-forward blocks,
  dropout, padding masks, and next-item candidate scoring.
- Smoke configs remain `reportable=false`.
- If PyTorch is unavailable, SASRec commands write clear `skipped_pytorch_unavailable` artifacts
  instead of falling back to Markov.

## Markov Status

The Markov transition ranker remains a smoke/pre-experiment baseline and must stay
`reportable=false`. It is not a SASRec or GRU4Rec substitute.

## Smoke Command

```bash
python scripts/train_sasrec.py --config configs/experiments/phase6_sasrec_smoke.yaml
```

No paper-scale sequential baseline results have been run.
