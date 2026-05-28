#!/usr/bin/env bash
# TGL-Rec Auto-Pipeline: waits for GPU, then runs full experiment sequence
# Deploy on server: nohup bash scripts/auto_pipeline.sh > outputs/auto_pipeline.log 2>&1 &
set -euo pipefail

PROJECT_DIR=~/projects/TGL-Rec
MODEL_PATH=/home/ajifang/models/Qwen/Qwen3-8B
PYTHON=/home/ajifang/miniconda3/envs/qwen_vllm/bin/python
export PYTHONPATH=$PROJECT_DIR/src

cd "$PROJECT_DIR"

echo "[auto] TGL-Rec Auto-Pipeline started at $(date)"
echo "[auto] Waiting for GPU to free up (Pony SRPD must finish)..."

# Phase 0: Wait for GPU (check every 5 min)
while true; do
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "[auto] $(date +%H:%M) GPU free: ${FREE_MB}MB"
    if [ "$FREE_MB" -gt 40000 ]; then
        echo "[auto] GPU FREE! (${FREE_MB}MB available). Starting experiments..."
        break
    fi
    sleep 300
done

# Phase 1: Observation (bf16, all domains)
echo ""
echo "============================================"
echo "[auto] Phase 1: OBSERVATION ($(date))"
echo "============================================"

DOMAINS=("beauty" "books" "electronics" "movies")

for domain in "${DOMAINS[@]}"; do
    DATA_DIR="data/domains/${domain}/same_candidate"
    if [ ! -f "${DATA_DIR}/ranking_valid.jsonl" ]; then
        echo "[obs-${domain}] SKIP: no data"
        continue
    fi

    OUT_DIR="outputs/observation/${domain}_bf16_smoke"
    rm -rf "$OUT_DIR"
    mkdir -p "$OUT_DIR"
    echo "[obs-${domain}] Running smoke (limit=20)..."
    $PYTHON scripts/run_observation.py \
        --domain "$domain" \
        --model-path "$MODEL_PATH" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUT_DIR" \
        --limit 20 \
        --variants base,shuffled,reversed,recent_only \
        --seed 42

    if [ -f "${OUT_DIR}/observation_summary.json" ]; then
        echo "[obs-${domain}] Results:"
        cat "${OUT_DIR}/observation_summary.json"
    else
        echo "[obs-${domain}] FAILED"
    fi
done

echo "[auto] Observation phase complete at $(date)"

# Phase 2: LoRA Training (all 4 domains)
echo ""
echo "============================================"
echo "[auto] Phase 2: LORA TRAINING ($(date))"
echo "============================================"

for domain in "${DOMAINS[@]}"; do
    LORA_DATA="outputs/lora_data/${domain}"
    LORA_OUT="outputs/lora_adapters/${domain}/seed_42"

    if [ ! -f "${LORA_DATA}/train.jsonl" ]; then
        echo "[lora-${domain}] SKIP: no training data"
        continue
    fi
    if [ -d "${LORA_OUT}/adapter" ]; then
        echo "[lora-${domain}] Already trained, skipping"
        continue
    fi

    echo "[lora-${domain}] Training LoRA adapter..."
    mkdir -p "$LORA_OUT"
    CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/train_lora_8b.py \
        --config configs/training/lora_8b.yaml \
        --data-dir "$LORA_DATA" \
        --base-model-path "$MODEL_PATH" \
        --output-dir "$LORA_OUT" \
        2>&1 | tee "${LORA_OUT}/train.log" || {
            echo "[lora-${domain}] FAILED"
            continue
        }
    echo "[lora-${domain}] Done at $(date)"
done

# Phase 3: Evaluation
echo ""
echo "============================================"
echo "[auto] Phase 3: EVALUATION ($(date))"
echo "============================================"

for domain in "${DOMAINS[@]}"; do
    LORA_OUT="outputs/lora_adapters/${domain}/seed_42"
    EVAL_OUT="outputs/evaluation/${domain}/seed_42"

    if [ ! -d "${LORA_OUT}/adapter" ]; then
        echo "[eval-${domain}] SKIP: no adapter"
        continue
    fi

    echo "[eval-${domain}] Running evaluation..."
    mkdir -p "$EVAL_OUT"
    CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/run_lora_rerank_eval.py \
        --config configs/experiments/week8_lora_8b_rerank_eval.yaml \
        --base-model-path "$MODEL_PATH" \
        --adapter-path "${LORA_OUT}/adapter" \
        --data-dir "data/domains/${domain}/same_candidate" \
        --output-dir "$EVAL_OUT" \
        2>&1 | tee "${EVAL_OUT}/eval.log" || {
            echo "[eval-${domain}] FAILED"
            continue
        }
done

echo ""
echo "============================================"
echo "[auto] ALL PHASES COMPLETE at $(date)"
echo "============================================"
