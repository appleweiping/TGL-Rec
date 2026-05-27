#!/usr/bin/env bash
# TGL-Rec Full Pipeline: Gate Training → LoRA Data → LoRA Training → Evaluation
# Run AFTER observation matrix confirms the research premise
set -euo pipefail

PROJECT_DIR=~/projects/TGL-Rec
MODEL_PATH=/home/ajifang/models/Qwen/Qwen3-8B
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd "$PROJECT_DIR"
conda activate qwen_vllm

echo "=== TGL-Rec Full Pipeline ==="
echo "Date: $(date)"
echo "Run ID: pipeline_${TIMESTAMP}"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

DOMAINS=("beauty" "books" "electronics" "movies")
SEEDS=(42 2026 2027 2028 2029)

# ============================================================
# Stage 1: Train Need-Gate (per domain, CPU-only, ~5min each)
# ============================================================
echo ""
echo "=== Stage 1: Need-Gate Training ==="

for domain in "${DOMAINS[@]}"; do
    GATE_OUT="outputs/gate_training/${domain}/seed_42"
    if [ -f "${GATE_OUT}/need_gate_weights.json" ]; then
        echo "[gate-${domain}] Already trained, skipping"
        continue
    fi

    CONFIG="configs/experiments/tglrec_gate_train_beauty.yaml"
    echo "[gate-${domain}] Training need-gate..."
    python -u scripts/train_need_gate.py \
        --config "$CONFIG" \
        --output-dir "$GATE_OUT" \
        --seed 42 \
        2>&1 | tee "${GATE_OUT}/train.log" || {
            echo "[gate-${domain}] FAILED - check ${GATE_OUT}/train.log"
            continue
        }
    echo "[gate-${domain}] Done: $(cat ${GATE_OUT}/gate_train_metrics.json 2>/dev/null || echo 'no metrics')"
done

# ============================================================
# Stage 2: Prepare LoRA Training Data (per domain, CPU-only)
# ============================================================
echo ""
echo "=== Stage 2: LoRA Data Preparation ==="

for domain in "${DOMAINS[@]}"; do
    GATE_WEIGHTS="outputs/gate_training/${domain}/seed_42/need_gate_weights.json"
    LORA_DATA="outputs/lora_data/${domain}"

    if [ ! -f "$GATE_WEIGHTS" ]; then
        echo "[lora-data-${domain}] SKIP: no gate weights"
        continue
    fi
    if [ -f "${LORA_DATA}/train.jsonl" ]; then
        echo "[lora-data-${domain}] Already prepared, skipping"
        continue
    fi

    echo "[lora-data-${domain}] Preparing LoRA training data..."
    python -u scripts/prepare_lora_data.py \
        --config "configs/experiments/tglrec_gate_train_beauty.yaml" \
        --gate-weights "$GATE_WEIGHTS" \
        --output-dir "$LORA_DATA" \
        --seed 42 \
        2>&1 | tee "${LORA_DATA}/prepare.log" || {
            echo "[lora-data-${domain}] FAILED"
            continue
        }
done

# ============================================================
# Stage 3: LoRA Training (GPU, ~2-4h per domain)
# ============================================================
echo ""
echo "=== Stage 3: LoRA Training ==="

for domain in "${DOMAINS[@]}"; do
    LORA_DATA="outputs/lora_data/${domain}"
    LORA_OUT="outputs/lora_adapters/${domain}/seed_42"

    if [ ! -f "${LORA_DATA}/train.jsonl" ]; then
        echo "[lora-train-${domain}] SKIP: no training data"
        continue
    fi
    if [ -d "${LORA_OUT}/adapter" ]; then
        echo "[lora-train-${domain}] Already trained, skipping"
        continue
    fi

    echo "[lora-train-${domain}] Training LoRA adapter..."
    mkdir -p "$LORA_OUT"
    CUDA_VISIBLE_DEVICES=0 python -u scripts/train_lora_8b.py \
        --config configs/training/lora_8b.yaml \
        --data-dir "$LORA_DATA" \
        --base-model-path "$MODEL_PATH" \
        --output-dir "$LORA_OUT" \
        2>&1 | tee "${LORA_OUT}/train.log" || {
            echo "[lora-train-${domain}] FAILED"
            continue
        }
    echo "[lora-train-${domain}] Done."
done

# ============================================================
# Stage 4: Evaluation (GPU, ~30min per domain)
# ============================================================
echo ""
echo "=== Stage 4: Evaluation ==="

for domain in "${DOMAINS[@]}"; do
    LORA_OUT="outputs/lora_adapters/${domain}/seed_42"
    EVAL_OUT="outputs/evaluation/${domain}/seed_42"

    if [ ! -d "${LORA_OUT}/adapter" ]; then
        echo "[eval-${domain}] SKIP: no adapter"
        continue
    fi

    echo "[eval-${domain}] Running evaluation..."
    mkdir -p "$EVAL_OUT"
    CUDA_VISIBLE_DEVICES=0 python -u scripts/run_lora_rerank_eval.py \
        --config configs/experiments/week8_lora_8b_rerank_eval.yaml \
        --base-model-path "$MODEL_PATH" \
        --adapter-path "${LORA_OUT}/adapter" \
        --data-dir "data/domains/${domain}/same_candidate" \
        --output-dir "$EVAL_OUT" \
        2>&1 | tee "${EVAL_OUT}/eval.log" || {
            echo "[eval-${domain}] FAILED"
            continue
        }
    echo "[eval-${domain}] Done."
done

echo ""
echo "=== Pipeline Complete ==="
echo "Results summary:"
for domain in "${DOMAINS[@]}"; do
    EVAL_OUT="outputs/evaluation/${domain}/seed_42"
    if [ -f "${EVAL_OUT}/metrics.json" ]; then
        echo "  ${domain}: $(python -c "import json; m=json.load(open('${EVAL_OUT}/metrics.json')); print(f'MRR={m.get(\"MRR\",0):.4f} HR@5={m.get(\"HR@5\",0):.4f} HR@10={m.get(\"HR@10\",0):.4f}')")"
    else
        echo "  ${domain}: NO RESULTS"
    fi
done
