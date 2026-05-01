# Pre-Experiment Checklist

- [x] Shared prediction schema.
- [x] Shared evaluator.
- [x] Config-driven split and candidate protocols.
- [x] Smoke train/checkpoint path.
- [x] LoRA/QLoRA dry-run plan without downloads.
- [x] Multi-seed smoke aggregation.
- [x] Significance-test interface.
- [x] Table export from metrics artifacts.
- [x] Experiment config validation.
- [x] Project validation command.
- [x] SASRec-style implementation path, with explicit PyTorch-unavailable skip behavior.
- [x] OursMethod design skeleton and ablation interfaces.
- [x] Lightweight TemporalGraphEncoder option, not full TGN.
- [x] Frozen protocol docs for datasets, splits, candidates, metrics, reportability, leakage.
- [x] NON_REPORTABLE pilot matrix and ablation pilot outputs.
- [x] Pilot resource estimate, failure audit, and table export.
- [ ] Real GRU4Rec/BERT4Rec/TiSASRec implementations.
- [ ] Protocol freeze before reportable runs.
- [ ] Paper-scale experiments.

Phase 6 smoke outputs are non-reportable. Markov remains smoke-only and must not be reported as
SASRec/GRU4Rec. If PyTorch is unavailable, SASRec and TemporalGraphEncoder smoke commands must
record `skipped_pytorch_unavailable`.
