# Server Runbook

This runbook is for continuing TGL-Rec on the shared server with minimal local
back-and-forth. It does not override the non-reportable gates.

Read `docs/codex_project_memory.md` before using or editing this runbook.

Codex cannot inspect or operate the shared server directly. The user runs these
commands and pastes logs/errors back. Future Codex agents must not infer server
success without pasted evidence, and must update this file whenever server
commands change.

Before long jobs:

- run `git pull` after Codex pushes local changes;
- inspect `nvidia-smi` and avoid killing unrelated processes blindly;
- preserve previous outputs with timestamped `mv` before reruns;
- add `test -f` or `test -d` guards before diagnostics that require generated
  files;
- keep private server configs, model checkpoints, PDFs, and output artifacts out
  of git.

## 1. Sync Code

```bash
cd ~/projects/TGL-Rec
git pull
conda activate qwen_vllm
```

## 2. Verify Immediate Phase 10 Diagnostic State Inherited From Phase 9E

The fixed-label-mask control adapters have reportedly trained successfully:

- `outputs/paper_runs/protocol_v1/lora_8b/history_only_sft/adapter`
- `outputs/paper_runs/protocol_v1/lora_8b/temporal_evidence_sft/adapter`

Before rerunning eval:

```bash
nvidia-smi
test -d outputs/paper_runs/protocol_v1/lora_8b/history_only_sft/adapter
test -d outputs/paper_runs/protocol_v1/lora_8b/temporal_evidence_sft/adapter
```

Run the small diagnostic only after GPU memory is clear:

```bash
if [ -d outputs/paper_runs/protocol_v1/lora_8b/eval ]; then
  mv outputs/paper_runs/protocol_v1/lora_8b/eval \
    outputs/paper_runs/protocol_v1/lora_8b/eval_before_phase10_$(date +%Y%m%d_%H%M%S)
fi

CUDA_VISIBLE_DEVICES=0 python -u scripts/run_lora_rerank_eval.py \
  --config configs/experiments/paper_lora_8b_rerank_eval.yaml \
  --base-model-path /home/ajifang/models/Qwen/Qwen3-8B \
  --limit 20 \
  --top-m 50

test -f outputs/paper_runs/protocol_v1/lora_8b/eval/predictions.jsonl

python scripts/diagnose_lora_predictions.py \
  --predictions outputs/paper_runs/protocol_v1/lora_8b/eval/predictions.jsonl \
  --output-dir outputs/paper_runs/protocol_v1/lora_8b/eval/diagnostics
```

Do not run diagnostics if `predictions.jsonl` does not exist.

## 3. Generate Four-Domain Plan

```bash
python scripts/plan_four_domain_runs.py \
  --external-root ~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks \
  --output outputs/plans/four_domain_server_plan.json \
  --shell-output outputs/plans/four_domain_server_runbook.sh
```

Inspect:

```bash
cat outputs/plans/four_domain_server_plan.json
sed -n '1,220p' outputs/plans/four_domain_server_runbook.sh
```

The plan should include `beauty`, `books`, `electronics`, and `movies` once all
external task directories exist.

The generated plan includes real commands only for implemented stages. It also
contains explicit `BLOCKED` lines for formal official-reference baselines and
TGL-Rec ablations that are not yet implemented. Do not replace those blocked
lines with `reference_*_sft` scaffolds.

The first new large-scale observation command is the base Qwen3-8B smoke run:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_lora_rerank_eval.py \
  --config configs/experiments/week8_qwen3_8b_base_observation.yaml \
  --base-model-path /home/ajifang/models/Qwen/Qwen3-8B \
  --limit 20
```

Only remove `--limit 20` after the resulting diagnostics show sane parse success
and candidate adherence.

Run lightweight local guard tests before launching long jobs:

```bash
python -m pytest \
  tests/unit/test_reference_baseline_configs.py \
  tests/unit/test_week8_lora_configs.py \
  tests/unit/test_four_domain_plan.py \
  tests/unit/test_run_compare.py -q
```

## 4. Import Frozen Same-Candidate Tasks

The generated runbook will call:

```bash
python scripts/import_week8_same_candidate.py \
  --task-dir <domain_valid_or_test_task> \
  --protocol-version protocol_week8_large10000_same_candidate
```

Rules:

- do not resample users;
- do not resample negatives;
- do not alter candidates;
- preserve `event_id/source_event_id`;
- keep `protocol_v1` intact.

## 5. Week8 LoRA Control Path

After import, build SFT data from train-only interactions:

```bash
python scripts/build_lora_sft_data.py \
  --config configs/experiments/week8_lora_8b_history_only.yaml \
  --materialize

python scripts/build_lora_sft_data.py \
  --config configs/experiments/week8_lora_8b_temporal_evidence.yaml \
  --materialize
```

The build step materializes one SFT directory per imported domain. Merge those
domain artifacts before training; the Week8 training configs point to these
`four_domain/<variant>` directories:

```bash
python scripts/merge_lora_sft_data.py \
  --input-root data/processed/lora_sft/protocol_week8_large10000_same_candidate \
  --output-dir data/processed/lora_sft/protocol_week8_large10000_same_candidate/four_domain/history_only_sft \
  --variant history_only_sft \
  --datasets beauty books electronics movies \
  --protocol-version protocol_week8_large10000_same_candidate

python scripts/merge_lora_sft_data.py \
  --input-root data/processed/lora_sft/protocol_week8_large10000_same_candidate \
  --output-dir data/processed/lora_sft/protocol_week8_large10000_same_candidate/four_domain/temporal_evidence_sft \
  --variant temporal_evidence_sft \
  --datasets beauty books electronics movies \
  --protocol-version protocol_week8_large10000_same_candidate
```

Check the merge manifests:

```bash
cat data/processed/lora_sft/protocol_week8_large10000_same_candidate/four_domain/history_only_sft/sft_merge_manifest.json
cat data/processed/lora_sft/protocol_week8_large10000_same_candidate/four_domain/temporal_evidence_sft/sft_merge_manifest.json
```

Then train adapters, adjusting private server configs only if needed:

```bash
mkdir -p \
  outputs/paper_runs/protocol_week8_large10000_same_candidate/lora_8b/history_only_sft \
  outputs/paper_runs/protocol_week8_large10000_same_candidate/lora_8b/temporal_evidence_sft

CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_lora_8b.py \
  --config configs/experiments/week8_lora_8b_history_only.yaml \
  > outputs/paper_runs/protocol_week8_large10000_same_candidate/lora_8b/history_only_sft/train.nohup.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_lora_8b.py \
  --config configs/experiments/week8_lora_8b_temporal_evidence.yaml \
  > outputs/paper_runs/protocol_week8_large10000_same_candidate/lora_8b/temporal_evidence_sft/train.nohup.log 2>&1 &
```

Evaluate on preserved external candidates:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_lora_rerank_eval.py \
  --config configs/experiments/week8_lora_8b_rerank_eval.yaml \
  --base-model-path /home/ajifang/models/Qwen/Qwen3-8B
```

`week8_lora_8b_rerank_eval.yaml` uses
`candidate_selection: preserve_external_candidates` and keeps
`do_not_merge_into_main_accuracy_table: true` until Week8 baselines exist under
the same protocol.

Once a baseline or our full framework run also emits shared prediction JSONL,
compare aligned events without mixing protocols:

```bash
python scripts/compare_prediction_runs.py \
  --run history_or_temporal=outputs/paper_runs/protocol_week8_large10000_same_candidate/lora_8b/eval/predictions.jsonl \
  --run baseline=outputs/paper_runs/protocol_week8_large10000_same_candidate/main_accuracy_seed0/predictions.jsonl \
  --baseline baseline \
  --allow-non-reportable \
  --output-dir outputs/paper_runs/protocol_week8_large10000_same_candidate/paired_compare/lora_vs_baseline
```

Omit `--allow-non-reportable` for paper-table comparisons. The comparison tool
will then reject scaffold/non-reportable rows, protocol mismatches, split
mismatches, and candidate-set mismatches.

## 6. Reportability Reminder

Current Week8 LoRA controls are needed for observation and framework debugging.
They are not a complete paper result until:

- four-domain artifacts are frozen and audited;
- official/fair baselines are implemented;
- our final framework ablations run;
- paired statistical tests are run on aligned event IDs;
- reviewer gate has no blocking issues.
