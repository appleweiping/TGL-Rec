# Phase 9E Local LoRA Limit-200 Audit

## Scope

This note records the server-side local LoRA reranking evaluation for `protocol_v1`
with `limit_per_dataset_adapter=200` and `top_m=50`.

The run used the trained Qwen3-8B LoRA adapters:

- `outputs/paper_runs/protocol_v1/lora_8b/history_only_sft/adapter`
- `outputs/paper_runs/protocol_v1/lora_8b/temporal_evidence_sft/adapter`

The evaluation artifacts were pulled back locally and stored at:

- `outputs/paper_runs/protocol_v1/lora_8b/eval_limit200/`
- `outputs/paper_runs/protocol_v1/lora_8b/eval_limit200.nohup.log`

The local `outputs/` tree is intentionally ignored by git.

## Commands Run

Server evaluation command:

```bash
cd ~/projects/TGL-Rec
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/run_lora_rerank_eval.py \
  --config configs/experiments/paper_lora_8b_rerank_eval.yaml \
  --base-model-path /home/ajifang/models/Qwen/Qwen3-8B \
  --limit 200 \
  --top-m 50 \
  > outputs/paper_runs/protocol_v1/lora_8b/eval_limit200.nohup.log 2>&1 &
```

Server packaging command:

```bash
cd ~/projects/TGL-Rec
tar -czf TGL-Rec-phase9e-lora-eval-limit200.tgz \
  outputs/paper_runs/protocol_v1/lora_8b/eval \
  outputs/paper_runs/protocol_v1/lora_8b/eval_limit200.nohup.log
```

Local placement:

```powershell
tar -xzf TGL-Rec-phase9e-lora-eval-limit200.tgz -C .codex\staging\phase9e_lora_eval_limit200
Copy-Item -Recurse .codex\staging\phase9e_lora_eval_limit200\outputs\paper_runs\protocol_v1\lora_8b\eval `
  outputs\paper_runs\protocol_v1\lora_8b\eval_limit200
Copy-Item .codex\staging\phase9e_lora_eval_limit200\outputs\paper_runs\protocol_v1\lora_8b\eval_limit200.nohup.log `
  outputs\paper_runs\protocol_v1\lora_8b\eval_limit200.nohup.log
```

## Observed Manifest

```json
{
  "base_model_path": "/home/ajifang/models/Qwen/Qwen3-8B",
  "candidate_limit": 50,
  "dry_run": false,
  "limit_per_dataset_adapter": 200,
  "num_predictions": 800,
  "runtime_seconds": 16567.82990919915,
  "split": "test",
  "status": "succeeded"
}
```

The predictions file contains 800 JSONL rows, matching the manifest.

## Metrics

| Dataset | Method | Recall@5 | Recall@10 | NDCG@10 | MRR@10 | Parse success | Candidate adherence | Validity | Hallucination |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `amazon_multidomain_filtered_iterative_k3` | `local_8b_lora::history_only_sft` | 0.005 | 0.005 | 0.005 | 0.005 | 0.285 | 1.000 | 1.000 | 0.000 |
| `amazon_multidomain_filtered_iterative_k3` | `local_8b_lora::temporal_evidence_sft` | 0.005 | 0.005 | 0.005 | 0.005 | 1.000 | 1.000 | 1.000 | 0.000 |
| `movielens_full` | `local_8b_lora::history_only_sft` | 0.345 | 0.345 | 0.345 | 0.345 | 0.745 | 1.000 | 1.000 | 0.000 |
| `movielens_full` | `local_8b_lora::temporal_evidence_sft` | 0.000 | 0.050 | 0.014585 | 0.005125 | 0.780 | 1.000 | 1.000 | 0.000 |

## Interpretation

The local LoRA training and inference pipeline is operational on real adapters.
The parser recovery change improved the ability to recover item IDs from non-JSON
outputs, and the temporal-evidence adapter is more format-stable on Amazon.

These results do not support a ranking-accuracy win for the temporal-evidence
adapter. On MovieLens, `history_only_sft` is substantially stronger in this
limit-200 diagnostic run. On Amazon, both adapters are near zero ranking quality
under this candidate protocol.

## Remaining Risks

- The LoRA adapters often emit bare IDs or prompt continuations instead of valid
  JSON, so parse success should be reported separately from ranking metrics.
- The run is diagnostic (`limit=200` per dataset-adapter), not a full paper-scale
  evaluation.
- The generation warnings indicate `temperature`, `top_p`, and `top_k` may be
  ignored by the current Transformers generation path. This should be cleaned up
  before reporting deterministic decoding settings.
- The temporal-evidence objective improves format adherence in some settings but
  does not currently improve ranking accuracy.

## Next Step

Stop scaling this LoRA variant as a paper-winning result unless a targeted
diagnostic identifies a correctable issue. The next useful engineering step is
to compare the LoRA outputs against the frozen candidate sets and training labels
to determine whether the poor ranking is caused by prompt continuation behavior,
adapter objective mismatch, or candidate construction.
