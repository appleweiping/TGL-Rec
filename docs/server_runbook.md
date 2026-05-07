# Server Runbook

This runbook is for continuing TGL-Rec on the shared server with minimal local
back-and-forth. It does not override the non-reportable gates.

## 1. Sync Code

```bash
cd ~/projects/TGL-Rec
git pull
conda activate qwen_vllm
```

## 2. Verify Immediate Phase 9E Diagnostic State

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

The build step materializes one SFT directory per imported domain. The current
`week8_lora_8b_*` training configs set `sft.data_dir` to the `books` directory
as a conservative server smoke path. For a true four-domain LoRA run, first add
or generate a merged train/valid SFT directory and point `sft.data_dir` to it, or
train one adapter per domain and evaluate them separately.

Then train adapters, adjusting private server configs only if needed:

```bash
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
`candidate_selection: preserve_external_candidates`.

## 6. Reportability Reminder

Current Week8 LoRA controls are needed for observation and framework debugging.
They are not a complete paper result until:

- four-domain artifacts are frozen and audited;
- official/fair baselines are implemented;
- our final framework ablations run;
- paired statistical tests are run on aligned event IDs;
- reviewer gate has no blocking issues.
