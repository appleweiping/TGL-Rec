# Codex Handoff: Phase 9E LoRA Baselines

Historical note: this handoff records Phase 9E provenance and should not be the
first-read active plan for new work. Future Codex threads should start with
`docs/codex_project_memory.md`, then `docs/phase10_master_plan.md` and
`docs/server_runbook.md`. If this file conflicts with those Phase 10 documents,
the Phase 10 documents win.

## Current State

Branch: `codex/phase9e-lora-rerank-eval`

Latest pushed commits:

- `5e31e05 Require official code for reference baselines`
- `29c0d48 Add reference baseline implementation cards`
- `22b3cc2 Map concrete reference methods for adaptation`
- `fe8391b Document algorithm-fidelity baseline principle`
- `f928f8b Clarify reference baseline fidelity requirements`
- `201de0e Add collaborative reference LoRA baseline`
- `7d38a31 Add Week8 same-candidate importer`
- `b2ee629 Document large same-candidate protocol handoff`
- `3eca9a0 Speed up LoRA SFT batching`
- `3f5ab54 Keep LoRA training on visible GPU`
- `1b9ed07 Add Phase 9E Codex handoff notes`

This repository is being used as a research-grade LLM4Rec framework. Do not
fabricate results, do not treat diagnostic runs as paper results, and do not
commit large artifacts, model outputs, PDFs, or server-specific configs.

## What Has Been Done

The Phase 9E local LoRA pipeline is operational:

- Qwen3-8B LoRA/QLoRA training runner exists in `scripts/train_lora_8b.py`.
- Local LoRA rerank evaluation runner exists in `scripts/run_lora_rerank_eval.py`.
- Prediction diagnostics exist in `scripts/diagnose_lora_predictions.py`.
- LoRA SFT training now masks prompt tokens and trains only assistant answer
  tokens, with optional EOS supervision.
- Local LoRA evaluation now uses deterministic sampled `top_m` candidates rather
  than taking the first catalog items and appending the target.
- The old limit-200 LoRA evaluation was pulled back locally and documented in
  `docs/phase9e_lora_limit200_audit.md`.
- Reference-paper scaffold slots were added in
  `src/llm4rec/trainers/sft_variants.py`.
- LoRA SFT training no longer pads every sample to `max_seq_length` during
  tokenization. It now uses a dynamic padding collator, and tracked 8B LoRA
  training templates use `max_seq_length: 1024`.
- Week8 large same-candidate task import support exists in
  `src/llm4rec/data/week8_same_candidate.py`, with the CLI wrapper
  `scripts/import_week8_same_candidate.py`. It preserves external event IDs,
  source event IDs, candidate sets, and split labels.

Registered SFT variants:

- `history_only_sft`: control group.
- `temporal_evidence_sft`: our observation/evidence variant.
- `reference_preference_sft`: reference baseline slot for preference alignment,
  controllability, and instruction preference papers.
- `reference_semantic_sft`: reference baseline slot for semantic/text/multimodal
  matching papers.
- `reference_long_tail_sft`: reference baseline slot for long-tail and
  popularity-bias papers.
- `reference_collaborative_sft`: reference baseline slot for item co-occurrence,
  neighborhood preference, and sequential collaborative-filtering papers.

Reference baseline plan:

- See `docs/reference_lora_baseline_plan.md`.
- See `docs/reference_baseline_fidelity.md` before describing any reference
  variant as a senior-recommended original baseline.
- See `docs/reference_method_adaptation_map.md` for the selected concrete
  reference methods to implement first.
- Reference PDFs are local research material under `references/NH/` and
  `references/NR/`.
- Do not commit PDFs or copied paper text.
- The intended design is to adapt reference paper methods into this framework,
  using the same Qwen3-8B base model, same splits, same candidate protocol,
  same prediction schema, and same evaluator. LoRA/adapter/head/loss/scoring
  logic must stay method-specific where the official algorithm requires it.
- Current `reference_*_sft` variants are candidate scaffolds until mapped to
  concrete reference papers/projects. Do not call them original baselines until
  that mapping and implementation are complete.

## Important User Direction

The user clarified that the supervisor/senior-student recommended baselines are
the paper projects inside the local `references/` folder, not merely traditional
recommender baselines.

All main baselines should be implemented through the project framework with the
Qwen3-8B base model where faithful. The goal is to compare our observation
against faithful reference-paper baselines fairly, and to test whether the
observed phenomenon also appears in other baselines rather than only in our
method.

Important baseline-fidelity principle: preserve each senior-recommended
baseline's own training and scoring logic as much as possible; unify the
experimental protocol instead. Shared controls are data, candidates, splits,
metrics, prediction schema, and Qwen3-8B base model. LoRA/adapter training,
extra heads, losses, and scoring logic should follow each official baseline
algorithm where faithful. Do not turn reference baselines into generic prompt
toys merely for uniformity.
Use official baseline code/projects whenever available. TGL-Rec should provide
data/protocol/base-model/prediction-schema adapters around official code, not
invent local lookalikes. If official code is not identified, keep that method
out of the main baseline set unless the user explicitly approves a labeled
non-official reproduction.

The observation has not been proven yet. The current status is: the hypothesis,
LoRA framework, and diagnostic tooling exist, but old adapter results were
diagnostic and affected by prompt-continuation behavior before the label-mask
fix. Do not describe the observation as validated until the fixed-label-mask
adapters and reference-paper baselines are trained and evaluated on frozen
candidate protocols.

The current datasets are still considered preliminary/toy-ish for final paper
claims. A stronger conference-grade dataset is being generated by another
project on the same server. Future work should integrate it as a new frozen
protocol version instead of overwriting existing `protocol_v1` results.

The new large-scale same-candidate protocol is being produced under:

```text
~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/
```

Known target domains are `books`, `electronics`, and `movies`, with up to 10,000
users per domain and 1 positive plus 100 popularity-sampled negatives per event.
The user also has a complete `beauty` domain on the server, to be integrated
once its exact layout is confirmed. See
`docs/week8_large_same_candidate_protocol.md` for the import rules and required
event/candidate preservation.

## Existing Data And Results

Current local/server protocol artifacts:

- `outputs/artifacts/protocol_v1/movielens_full/`
- `outputs/artifacts/protocol_v1/amazon_multidomain_filtered_iterative_k3/`
- `data/processed/lora_sft/protocol_v1/`

Old trained adapters, before the label-mask fix:

- `outputs/paper_runs/protocol_v1/lora_8b/history_only_sft/adapter`
- `outputs/paper_runs/protocol_v1/lora_8b/temporal_evidence_sft/adapter`

Old limit-200 diagnostic evaluation:

- `outputs/paper_runs/protocol_v1/lora_8b/eval_limit200/`
- `outputs/paper_runs/protocol_v1/lora_8b/eval_limit200.nohup.log`

Old limit-200 key finding:

- Pipeline works, but the result should not be treated as a paper win.
- Prompt continuation was common, especially for MovieLens outputs.
- This motivated the SFT label-mask fix.

## Immediate Next Step

Server status reported by the user:

- `history_only_sft` fixed-label-mask fast training succeeded in about 4h47m.
- `temporal_evidence_sft` fixed-label-mask fast training succeeded in about
  4h50m.
- A small post-fix eval attempt failed with CUDA OOM because other processes were
  occupying GPU 0. No `predictions.jsonl` was produced for that failed attempt.

The next Codex should help the user clear GPU memory and run the small
post-label-mask diagnostic eval, not retrain these two adapters again unless a
new issue is found.

On the server, use:

```bash
cd ~/projects/TGL-Rec
git remote set-url origin git@github.com:appleweiping/TGL-Rec.git
git pull
```

If an old training process was started before commit `3eca9a0`, stop and restart
it; running Python processes do not pick up the dynamic-padding change. Server
private configs are ignored by git, so after pulling check:

```bash
grep -n "max_seq_length" configs/experiments/server_lora_8b_*.yaml
```

Set the server private LoRA training configs to `max_seq_length: 1024` before
restarting if they still say `2048`.

Run the small diagnostic eval:

```bash
cd ~/projects/TGL-Rec
conda activate qwen_vllm

mv outputs/paper_runs/protocol_v1/lora_8b/eval \
  outputs/paper_runs/protocol_v1/lora_8b/eval_before_labelmask_fast_$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=0 python -u scripts/run_lora_rerank_eval.py \
  --config configs/experiments/paper_lora_8b_rerank_eval.yaml \
  --base-model-path /home/ajifang/models/Qwen/Qwen3-8B \
  --limit 20 \
  --top-m 50

python scripts/diagnose_lora_predictions.py \
  --predictions outputs/paper_runs/protocol_v1/lora_8b/eval/predictions.jsonl \
  --output-dir outputs/paper_runs/protocol_v1/lora_8b/eval/diagnostics
```

Only proceed to `--limit 200` if prompt-continuation rate drops and parse
success remains stable.

If eval fails with CUDA OOM, inspect `nvidia-smi` and stop unrelated processes
before rerunning. Do not run diagnostics unless
`outputs/paper_runs/protocol_v1/lora_8b/eval/predictions.jsonl` exists.

## Files Not To Commit

Leave these local/untracked unless explicitly requested:

- `TGL-Rec-*.tgz`
- `TGL_Rec_Project_Experiment_Plan_CN_EN.md`
- `configs/experiments/server_lora_8b_history_only.yaml`
- `configs/experiments/server_lora_8b_temporal_evidence.yaml`
- `references/**/*.pdf`
- `references/**/*.zip`
- `outputs/`
- `data/processed/`

Server configs include machine-local paths such as
`/home/ajifang/models/Qwen/Qwen3-8B`; keep them local unless a sanitized template
is created.

## Recommended Follow-Up After Retraining

After the two control adapters are retrained and diagnostics look sane:

1. Pull results back locally with tar/scp.
2. Update the Phase 9E audit with fixed-label-mask metrics.
3. Index the `references/` papers into lightweight notes.
4. Map each selected paper to one LoRA variant family or add a new variant only
   when the method requires a genuinely distinct signal.
5. Use `scripts/import_week8_same_candidate.py` to import the Week8 large
   same-candidate protocol from `pony-rec-rescue-shadow-v6`, preserving
   `event_id/source_event_id`, `user_id`, `item_id`, `split`, and exact
   candidate sets.
6. Build/train/evaluate faithful reference-paper baselines under the same
   framework on the frozen large same-candidate protocol.
