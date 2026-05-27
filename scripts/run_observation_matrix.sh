#!/usr/bin/env bash
# TGL-Rec Observation Matrix Runner
# Proves the research premise: base Qwen3-8B ignores temporal order
set -euo pipefail

PROJECT_DIR=~/projects/TGL-Rec
MODEL_PATH=/home/ajifang/models/Qwen/Qwen3-8B
OUTPUT_ROOT=outputs/observation

cd "$PROJECT_DIR"
conda activate qwen_vllm

echo "=== TGL-Rec Observation Matrix ==="
echo "Date: $(date)"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

DOMAINS=("beauty" "books" "electronics" "movies")
VARIANTS="base,shuffled,reversed,recent_only,no_time"

for domain in "${DOMAINS[@]}"; do
    DATA_DIR="data/domains/${domain}/same_candidate"
    OUT_DIR="${OUTPUT_ROOT}/${domain}"

    if [ ! -f "${DATA_DIR}/ranking_valid.jsonl" ]; then
        echo "[SKIP] ${domain}: no ranking_valid.jsonl found"
        continue
    fi

    echo ""
    echo "=== Domain: ${domain} ==="

    # Phase 1: Smoke test (limit=20)
    echo "[obs-${domain}] Phase 1: Smoke test (limit=20)..."
    CUDA_VISIBLE_DEVICES=0 python -u scripts/run_observation.py \
        --domain "$domain" \
        --model-path "$MODEL_PATH" \
        --data-dir "$DATA_DIR" \
        --output-dir "${OUT_DIR}/smoke" \
        --limit 20 \
        --variants "$VARIANTS" \
        --seed 42

    # Check smoke results before full run
    if [ -f "${OUT_DIR}/smoke/observation_summary.json" ]; then
        echo "[obs-${domain}] Smoke results:"
        python -c "
import json
with open('${OUT_DIR}/smoke/observation_summary.json') as f:
    d = json.load(f)
for v, m in d.items():
    print(f'  {v}: MRR={m.get(\"MRR\",0):.4f} HR@5={m.get(\"HR@5\",0):.4f} parse={m.get(\"parse_success_rate\",0):.2f}')
"
        # Phase 2: Full run (no limit)
        echo "[obs-${domain}] Phase 2: Full run..."
        CUDA_VISIBLE_DEVICES=0 python -u scripts/run_observation.py \
            --domain "$domain" \
            --model-path "$MODEL_PATH" \
            --data-dir "$DATA_DIR" \
            --output-dir "${OUT_DIR}/full" \
            --variants "$VARIANTS" \
            --seed 42
    else
        echo "[obs-${domain}] ERROR: Smoke test failed, skipping full run"
    fi
done

echo ""
echo "=== Observation Matrix Complete ==="
echo "Results in: ${OUTPUT_ROOT}/"
echo "Next: bash scripts/run_gate_training.sh"
