# CC-PACE LoRA — Runbook (beauty judge fine-tuning)

Server: `pony-rec-gpu` (RTX 4090, 48.5GB). Repo `~/projects/TGL-Rec`. Env `tglrec-lora`
(torch 2.8.0+cu128, transformers 5.7.0, peft 0.19.1). Model `/home/ajifang/models/Qwen/Qwen3-8B`.

## What this is
LoRA fine-tune of the Qwen3-8B forced-choice judge with the listwise **Plackett-Luce top-1**
loss (+ dCor popularity penalty) over beauty TRAIN panels, to lift CC-PACE beauty NDCG@10 from the
zero-shot **0.1200** above the promax SOTA bar **0.1506**.

## Bug fixed before this could run
`scripts/train_cc_pace_lora.py` had a broken single-digit-token surrogate: it asserted the 2nd token
of `[NNN]` labels is unique per label. Qwen3 tokenizes every digit separately, so no single token
position discriminates 101 labels (`AssertionError: digit tokens must be unique per label`).
Memory wall: a real ~26-27k-token panel only fits ONE forward+backward on the 4090 (~50GB peak;
replicating the prompt per-label OOMs). Fix = score all 101 labels at ONE prompt-forward's last
position using **single-token, vocab-unique labels** (A,B,..,AA,..). Implemented via a new optional
`label_vocab` param on `schema.render_panel` (default `[NNN]` unchanged → inference `hf_judge.py`
untouched, all 10 CPU unit tests pass). Also added an OOM-skip guard (skip a rare oversized panel
instead of crashing; no fidelity loss). Edited files: `scripts/train_cc_pace_lora.py`,
`src/llm4rec/methods/cc_pace/schema.py` (staged in local repo branch `feat/cc-pace-vllm-judge`,
NOT committed).

## Cost (measured)
~22s / panel-step (1 fwd+bwd, bf16, grad-ckpt, LoRA r16). FULL ≈ fold-A(~778) × 2 epochs ≈ 1556
steps ≈ **~9.5-10 GPU-hours** — within the 24-48h budget. Pilot (192 steps) wall = 1:12:58, rc=0,
0 OOM.

## Pilot result (HEALTHY → GO)
T-normalized loss (loss×T) dropped 5.19→2.96→2.31 across epoch 1 (epoch-2 noisier under T→0.13
anneal on a 96-panel subset, expected). Valid rank-16 adapter saved.

## FULL run (launched detached)
```bash
# self-gating, crash-safe (setsid+nohup), full fidelity (--max-panels 0, epochs 2, rank 16)
bash scripts/launch_cc_pace_lora_full.sh      # writes outputs/cc_pace_beauty/lora_full/
```
Check progress: `tail -f outputs/cc_pace_beauty/lora_full/full_*.log` (loss row every 25 steps).
Adapter on success: `outputs/cc_pace_beauty/lora_full/adapter_model.safetensors`.

## EVAL the LoRA'd judge → GO/SOTA check (after FULL finishes)
```bash
bash scripts/eval_cc_pace_lora.sh
```
This (1) backs up the zero-shot `full.json`, (2) runs `scripts/cc_pace_beauty.py --variant full
--adapter outputs/cc_pace_beauty/lora_full` on the **973 beauty test** panels at full fidelity
(n_label_randomizations=4, fixed CF + profiles), (3) runs `scripts/cc_pace_go_verdict.py` → writes
`go_verdict_lora.json`. **STRONG GO / reportable** = post-LoRA NDCG@10 ≥ 0.1506 with paired-bootstrap
p<0.05, full > text_only (CF-token gap), panel-corruption drop ≥30%.

## Item 6 — preserved server commit
Server commit `44f755b0` (HEAD of `feat/cc-pace-vllm-judge`, couldn't push: no server git creds) is
saved as a patch: server `~/0001-test-cc-pace-decouple-vLLM-HF-judge-equivalence-via-.patch` and local
`D:\Research\TGL-Rec\cc_pace_lora\0001-test-cc-pace-decouple-vLLM-HF-judge-equivalence.patch` (21625B).
Apply locally with `git am < <patch>` if you want it in the local repo history.

## Local evidence
`D:\Research\TGL-Rec\cc_pace_lora\` : fixed `schema.py` + `train_cc_pace_lora.py`, `pilot/` (meta +
training_log + log + fold_assignment), the preserved patch, this runbook.
