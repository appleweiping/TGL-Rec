#!/usr/bin/env bash
# RW-PMI beauty kill test — server runner. Launches ONLY when GPU has enough free memory.
# Usage: bash run_rwpmi_beauty.sh [limit]   (limit=0 -> all 973 users; small int -> sanity)
set -u
PROJ=/home/ajifang/projects/pony-rec-rescue-shadow-v6
PY=/home/ajifang/miniconda3/envs/qwen_vllm/bin/python
TASK="$PROJ/outputs/baselines/external_tasks/beauty_supplementary_smallerN_100neg_test_same_candidate/ranking_test.jsonl"
OUT="$PROJ/outputs/rwpmi_beauty/zeroshot_metrics.json"
SCRIPT="$PROJ/scripts/rwpmi_zeroshot_beauty.py"
LIMIT="${1:-0}"

need_mb=17000
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "GPU free: ${free_mb} MiB (need >= ${need_mb})"
if [ "$free_mb" -lt "$need_mb" ]; then
  echo "INSUFFICIENT GPU — not launching. (another job is using the card)"
  exit 3
fi

cd "$PROJ" || exit 1
echo "launching RW-PMI beauty kill test (limit=$LIMIT) ..."
"$PY" "$SCRIPT" --task "$TASK" --out "$OUT" --limit "$LIMIT" --alpha 1.0 --delta 0.0
echo "done -> $OUT"
