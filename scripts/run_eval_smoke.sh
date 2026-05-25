#!/usr/bin/env bash
# Fast FP eval with vLLM (after training completes).
#
# Requires vLLM cu129 wheel (matches torch cu128). See requirements-exp.txt.
# Fallback: BACKEND=hf BATCH_SIZE=32 bash scripts/run_eval_smoke.sh 0
#
# Usage:
#   bash scripts/run_eval_smoke.sh
#   bash scripts/run_eval_smoke.sh 0          # GPU 0
#   CUDA_VISIBLE_DEVICES=2 bash scripts/run_eval_smoke.sh
#   BACKEND=hf BATCH_SIZE=32 bash scripts/run_eval_smoke.sh
#   GPU_MEM_UTIL=0.70 bash scripts/run_eval_smoke.sh 0
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source env.sh
source .venv/bin/activate

MODEL="${MODEL:-qwen3-8b}"
TASK="${TASK:-gsm8k}"
OPTIMIZER="${OPTIMIZER:-adamw}"
SEED="${SEED:-42}"
ADAPTATION="${ADAPTATION:-full_ft}"
STAGE="${STAGE:-fp}"
BACKEND="${BACKEND:-}"          # default from configs/eval.yaml (vllm)
BATCH_SIZE="${BATCH_SIZE:-}"    # default: auto for vllm
GPU_IDS="${1:-${GPU_IDS:-0}}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-}"

CMD=(
  python -m exp.eval.run_lm_eval
  --model "$MODEL"
  --task "$TASK"
  --optimizer "$OPTIMIZER"
  --seed "$SEED"
  --adaptation "$ADAPTATION"
  --stage "$STAGE"
  --gpu-ids "$GPU_IDS"
  --tensor-parallel-size "$TP_SIZE"
)

if [[ -n "$BACKEND" ]]; then
  CMD+=(--backend "$BACKEND")
fi
if [[ -n "$BATCH_SIZE" ]]; then
  CMD+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "$GPU_MEM_UTIL" ]]; then
  CMD+=(--gpu-memory-utilization "$GPU_MEM_UTIL")
fi

echo "=== FP eval (vLLM accelerated) ==="
echo "+ ${CMD[*]}"
exec "${CMD[@]}"
