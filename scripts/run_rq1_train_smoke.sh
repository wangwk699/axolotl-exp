#!/usr/bin/env bash
# RQ1 phase A smoke: Full-FT train -> FP lm-eval -> aggregate FP scores.
#
# Train tasks: sst2, rte, boolq, metamath, codefeedback
# Eval mapping (tasks.yaml eval_task): metamath->gsm8k, codefeedback->humaneval
#
# Prepare data: python -m exp.data.prepare_all
#
# Usage:
#   bash scripts/run_rq1_train_smoke.sh
#   bash scripts/run_rq1_train_smoke.sh 2              # 2 GPUs (default ids 0,1)
#   bash scripts/run_rq1_train_smoke.sh 3 0,1,2
#   bash scripts/run_rq1_train_smoke.sh 4 4,5,6,7
#
# Examples:
#   MODEL=qwen3-8b TASK=metamath bash scripts/run_rq1_train_smoke.sh 2
#   MODEL=qwen3-1.7b TASK=sst2 bash scripts/run_rq1_train_smoke.sh 1
#
# Debug pipeline (attach localhost:6001):
#   DEBUG=1 bash scripts/run_rq1_train_smoke.sh 2
#
# Lighter train+auto-eval without aggregate: scripts/run_train_smoke.sh
# Phase B (PTQ): scripts/run_rq1_ptq_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source env.sh
source .venv/bin/activate

NUM_GPUS="${1:-${NUM_GPUS:-2}}"
GPU_IDS="${2:-${GPU_IDS:-}}"

MODEL="${MODEL:-qwen3-8b}"
TASK="${TASK:-metamath}"
OPTIMIZER="${OPTIMIZER:-adamw}"
SEED="${SEED:-42}"
NUM_TRAIN="${NUM_TRAIN:-1000}"
NUM_EVAL="${NUM_EVAL:-1000}"

if [[ -z "$GPU_IDS" ]]; then
  GPU_IDS="$(python3 - <<PY
n = int("${NUM_GPUS}")
print(",".join(str(i) for i in range(n)))
PY
)"
fi

ARGS=(
  --model "$MODEL"
  --task "$TASK"
  --optimizer "$OPTIMIZER"
  --seed "$SEED"
  --num-train "$NUM_TRAIN"
  --num-eval "$NUM_EVAL"
  --num-gpus "$NUM_GPUS"
  --gpu-ids "$GPU_IDS"
)

if [[ "${DEBUG:-0}" == "1" ]]; then
  DEBUG_PORT="${DEBUG_PORT:-6001}"
  CMD=(
    python -m debugpy --listen "$DEBUG_PORT" --wait-for-client
    -m exp.pipeline.run_rq1_train
    "${ARGS[@]}"
  )
else
  CMD=(python -m exp.pipeline.run_rq1_train "${ARGS[@]}")
fi

echo "=== RQ1 train phase ==="
echo "  model=$MODEL task=$TASK optimizer=$OPTIMIZER seed=$SEED"
echo "  num_gpus=$NUM_GPUS gpu_ids=$GPU_IDS debug=${DEBUG:-0}"
echo "+ ${CMD[*]}"
exec "${CMD[@]}"
