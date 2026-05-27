#!/usr/bin/env bash
# TGL-Rec Server Setup Script
# Run on: pony-rec-gpu (125.71.97.70:15302, user ajifang)
# Prerequisites: conda env 'qwen_vllm' with torch, transformers, peft
set -euo pipefail

PROJECT_DIR=~/projects/TGL-Rec
PONY_TASKS=~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks
MODEL_PATH=/home/ajifang/models/Qwen/Qwen3-8B

echo "=== TGL-Rec Server Setup ==="
echo "Date: $(date)"

# Step 1: Clone or pull
if [ -d "$PROJECT_DIR" ]; then
    echo "[1/5] Pulling latest code..."
    cd "$PROJECT_DIR"
    git pull
else
    echo "[1/5] Cloning repository..."
    cd ~/projects
    git clone git@github.com:appleweiping/TGL-Rec.git
    cd "$PROJECT_DIR"
fi

# Step 2: Activate environment and install
echo "[2/5] Setting up environment..."
conda activate qwen_vllm
pip install -e '.[models]' --quiet 2>/dev/null || pip install -e '.[models]'

# Step 3: Verify model access
echo "[3/5] Verifying model access..."
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi
echo "  Model OK: $MODEL_PATH"

# Step 4: Verify Pony external tasks exist
echo "[4/5] Verifying Pony external task directories..."
DOMAINS=("beauty_supplementary_smallerN_100neg" "books_large10000_100neg" "electronics_large10000_100neg" "movies_large10000_100neg")
SPLITS=("valid" "test")
MISSING=0
for domain in "${DOMAINS[@]}"; do
    for split in "${SPLITS[@]}"; do
        task_dir="${PONY_TASKS}/${domain}_${split}_same_candidate"
        if [ -d "$task_dir" ]; then
            echo "  OK: ${domain}_${split}"
        else
            echo "  MISSING: $task_dir"
            MISSING=$((MISSING + 1))
        fi
    done
done
if [ $MISSING -gt 0 ]; then
    echo "WARNING: $MISSING task directories missing. Some domains may not be importable."
fi

# Step 5: Import frozen same-candidate tasks
echo "[5/5] Importing frozen same-candidate tasks..."
mkdir -p outputs/artifacts
for domain in "${DOMAINS[@]}"; do
    for split in "${SPLITS[@]}"; do
        task_dir="${PONY_TASKS}/${domain}_${split}_same_candidate"
        if [ -d "$task_dir" ]; then
            echo "  Importing: ${domain}_${split}..."
            python scripts/import_week8_same_candidate.py \
                --task-dir "$task_dir" \
                --protocol-version protocol_week8_large10000_same_candidate \
                2>&1 | tail -3
        fi
    done
done

# Step 6: Prepare domain data for gate training
echo "[6/6] Preparing domain data for gate training..."
python scripts/prepare_domain_data.py \
    --artifacts-root outputs/artifacts/protocol_week8_large10000_same_candidate \
    --output-root data/domains \
    --domains beauty books electronics movies

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. Run observation: bash scripts/run_observation_matrix.sh"
echo "  2. Train gate: bash scripts/run_gate_training.sh"
echo "  3. Full pipeline: bash scripts/run_full_pipeline.sh"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || true
