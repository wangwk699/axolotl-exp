#!/usr/bin/env bash
# RQ1 phase B smoke: PTQ -> quant lm-eval -> aggregate FP vs quant.
#
# Requires phase A complete (FP checkpoint under artifacts/.../full_ft/fp).
# Run phase A first: bash scripts/run_rq1_train_smoke.sh
#
# Usage:
#   bash scripts/run_rq1_ptq_smoke.sh
#   bash scripts/run_rq1_ptq_smoke.sh 0          # eval on GPU 0
#   TASK=metamath bash scripts/run_rq1_ptq_smoke.sh 0
#
# Skip quantization (only eval existing PTQ dirs + aggregate):
#   SKIP_PTQ=1 bash scripts/run_rq1_ptq_smoke.sh 0
#
# Debug pipeline:
#   DEBUG=1 bash scripts/run_rq1_ptq_smoke.sh 0
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source env.sh
source .venv/bin/activate

GPU_IDS="${1:-${GPU_IDS:-0}}"

MODEL="${MODEL:-qwen3-8b}"
TASK="${TASK:-metamath}"
OPTIMIZER="${OPTIMIZER:-adamw}"
SEED="${SEED:-42}"
SKIP_PTQ="${SKIP_PTQ:-0}"

ARGS=(
  --model "$MODEL"
  --task "$TASK"
  --optimizer "$OPTIMIZER"
  --seed "$SEED"
  --gpu-ids "$GPU_IDS"
)

if [[ "$SKIP_PTQ" == "1" ]]; then
  ARGS+=(--skip-ptq)
fi

if [[ -n "${EVAL_BACKEND:-}" ]]; then
  ARGS+=(--eval-backend "$EVAL_BACKEND")
fi
if [[ -n "${EVAL_BATCH_SIZE:-}" ]]; then
  ARGS+=(--eval-batch-size "$EVAL_BATCH_SIZE")
fi

if [[ "${DEBUG:-0}" == "1" ]]; then
  DEBUG_PORT="${DEBUG_PORT:-6001}"
  CMD=(
    python -m debugpy --listen "$DEBUG_PORT" --wait-for-client
    -m exp.pipeline.run_rq1_ptq
    "${ARGS[@]}"
  )
else
  CMD=(python -m exp.pipeline.run_rq1_ptq "${ARGS[@]}")
fi

echo "=== RQ1 PTQ phase ==="
echo "  model=$MODEL task=$TASK optimizer=$OPTIMIZER seed=$SEED"
echo "  gpu_ids=$GPU_IDS skip_ptq=$SKIP_PTQ debug=${DEBUG:-0}"
echo "+ ${CMD[*]}"
exec "${CMD[@]}"
