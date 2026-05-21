#!/bin/bash
# TGL-Rec Server Deployment Script
# This deploys TGL-Rec as an INDEPENDENT project on the shared server.
# It does NOT touch the Pony project in any way.
#
# Server: pony-rec-gpu (125.71.97.70:15302, user: ajifang)
# TGL-Rec path: ~/projects/TGL-Rec/ (separate from ~/projects/pony-rec-rescue-shadow-v6/)
#
# Usage: Run each section manually on the server. Do NOT run this as a single script.

set -e

# ============================================================
# SECTION 1: Clone and setup (run once)
# ============================================================

echo "=== TGL-Rec Independent Deployment ==="
echo "This project is SEPARATE from Pony. It only READS Pony's baseline scores."

cd ~/projects/

# Clone TGL-Rec (if not already present)
if [ ! -d "TGL-Rec" ]; then
    git clone https://github.com/appleweiping/TGL-Rec.git
    cd TGL-Rec
else
    cd TGL-Rec
    git pull
fi

echo "TGL-Rec directory: $(pwd)"
echo "Pony directory (READ-ONLY reference): ~/projects/pony-rec-rescue-shadow-v6/"

# ============================================================
# SECTION 2: Environment setup
# ============================================================

# Create dedicated conda env (separate from Pony's qwen_vllm)
if ! conda env list | grep -q "tglrec"; then
    conda create -n tglrec python=3.11 -y
fi
conda activate tglrec

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.40 peft>=0.10 datasets accelerate
pip install scipy numpy pandas scikit-learn
pip install pyyaml tqdm

# Install TGL-Rec in editable mode
pip install -e .

echo "Environment ready: $(python --version), torch=$(python -c 'import torch; print(torch.__version__)')"

# ============================================================
# SECTION 3: Data symlinks (READ-ONLY from Pony external tasks)
# ============================================================

# We symlink Pony's frozen external tasks (read-only) into our data directory.
# This avoids copying large files while maintaining independence.

PONY_EXTERNAL="$HOME/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks"
TGLREC_DATA="$HOME/projects/TGL-Rec/data"

mkdir -p "$TGLREC_DATA"

# Beauty
if [ -d "$PONY_EXTERNAL/beauty_supplementary_smallerN_100neg_valid_same_candidate" ]; then
    ln -sfn "$PONY_EXTERNAL/beauty_supplementary_smallerN_100neg_valid_same_candidate" "$TGLREC_DATA/beauty_valid"
    ln -sfn "$PONY_EXTERNAL/beauty_supplementary_smallerN_100neg_test_same_candidate" "$TGLREC_DATA/beauty_test"
    echo "✓ Beauty linked"
fi

# Books
if [ -d "$PONY_EXTERNAL/books_large10000_100neg_valid_same_candidate" ]; then
    ln -sfn "$PONY_EXTERNAL/books_large10000_100neg_valid_same_candidate" "$TGLREC_DATA/books_valid"
    ln -sfn "$PONY_EXTERNAL/books_large10000_100neg_test_same_candidate" "$TGLREC_DATA/books_test"
    echo "✓ Books linked"
fi

# Electronics
if [ -d "$PONY_EXTERNAL/electronics_large10000_100neg_valid_same_candidate" ]; then
    ln -sfn "$PONY_EXTERNAL/electronics_large10000_100neg_valid_same_candidate" "$TGLREC_DATA/electronics_valid"
    ln -sfn "$PONY_EXTERNAL/electronics_large10000_100neg_test_same_candidate" "$TGLREC_DATA/electronics_test"
    echo "✓ Electronics linked"
fi

# Movies
if [ -d "$PONY_EXTERNAL/movies_large10000_100neg_valid_same_candidate" ]; then
    ln -sfn "$PONY_EXTERNAL/movies_large10000_100neg_valid_same_candidate" "$TGLREC_DATA/movies_valid"
    ln -sfn "$PONY_EXTERNAL/movies_large10000_100neg_test_same_candidate" "$TGLREC_DATA/movies_test"
    echo "✓ Movies linked"
fi

echo ""
echo "Data layout:"
ls -la "$TGLREC_DATA/"

# ============================================================
# SECTION 4: Verify GPU and model access
# ============================================================

nvidia-smi
echo ""
echo "Model path check:"
test -d /home/ajifang/models/Qwen/Qwen3-8B && echo "✓ Qwen3-8B available" || echo "✗ Qwen3-8B NOT found"

# ============================================================
# SECTION 5: Run observation experiment (Block 1)
# ============================================================

# This is the FIRST experiment: does base Qwen3-8B ignore temporal order?
# Run AFTER sections 1-4 are confirmed working.

echo "=== Observation Experiment ==="
echo "Testing if base Qwen3-8B is sensitive to history order..."

# Small diagnostic first (limit=20)
CUDA_VISIBLE_DEVICES=0 python scripts/run_observation.py \
    --domain beauty \
    --model-path /home/ajifang/models/Qwen/Qwen3-8B \
    --data-dir data/beauty_valid \
    --output-dir outputs/observation/beauty/ \
    --limit 20 \
    --variants base,shuffled,reversed,recent_only

echo "Check outputs/observation/beauty/ for results before scaling up."
