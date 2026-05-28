#!/usr/bin/env bash
# Smoke: one-command Full-FT train + auto lm-eval (exp.finetune.run_axolotl).
#
# Train tasks: sst2, rte, boolq, metamath, codefeedback (see configs/tasks.yaml).
# Auto-eval uses eval_task: metamath->gsm8k, codefeedback->humaneval, others->self.
#
# Prepare data once:
#   python -m exp.data.prepare_all
#
# Usage:
#   bash scripts/run_train_smoke.sh
#   bash scripts/run_train_smoke.sh 2              # 2 GPUs (default ids 0,1)
#   bash scripts/run_train_smoke.sh 3 0,1,2        # 3 GPUs on ids 0,1,2
#   bash scripts/run_train_smoke.sh 4 4,5,6,7      # 4 GPUs on ids 4,5,6,7
#
# Examples:
#   MODEL=qwen3-8b TASK=metamath bash scripts/run_train_smoke.sh 2
#   MODEL=qwen3-8b TASK=codefeedback bash scripts/run_train_smoke.sh 2
#   MODEL=qwen3-1.7b TASK=sst2 bash scripts/run_train_smoke.sh 1
#
# Train only (no auto-eval):
#   NO_AUTO_EVAL=1 bash scripts/run_train_smoke.sh 2
#
# RQ1 train phase (train + FP eval + aggregate): scripts/run_rq1_train_smoke.sh
# Debug in-process train: scripts/debug_axolotl_train.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source env.sh
source .venv/bin/activate

NUM_GPUS="${1:-${NUM_GPUS:-1}}"
GPU_IDS="${2:-${GPU_IDS:-}}"

MODEL="${MODEL:-qwen3-8b}"
TASK="${TASK:-metamath}"
OPTIMIZER="${OPTIMIZER:-adamw}"
SEED="${SEED:-42}"
RQ="${RQ:-1}"
ADAPTATION="${ADAPTATION:-full_ft}"
NO_AUTO_EVAL="${NO_AUTO_EVAL:-0}"
NUM_TRAIN="${NUM_TRAIN:-1000}"
NUM_EVAL="${NUM_EVAL:-1000}"

if [[ -z "$GPU_IDS" ]]; then
  GPU_IDS="$(python3 - <<PY
n = int("${NUM_GPUS}")
print(",".join(str(i) for i in range(n)))
PY
)"
fi

TRAIN_ARGS=(
  --rq "$RQ"
  --adaptation "$ADAPTATION"
  --model "$MODEL"
  --task "$TASK"
  --optimizer "$OPTIMIZER"
  --seed "$SEED"
  --num-train "$NUM_TRAIN"
  --num-eval "$NUM_EVAL"
  --num-gpus "$NUM_GPUS"
  --gpu-ids "$GPU_IDS"
)

if [[ "$NO_AUTO_EVAL" == "1" ]]; then
  TRAIN_ARGS+=(--no-auto-eval)
fi

if [[ -n "${EVAL_BACKEND:-}" ]]; then
  TRAIN_ARGS+=(--eval-backend "$EVAL_BACKEND")
fi
if [[ -n "${EVAL_BATCH_SIZE:-}" ]]; then
  TRAIN_ARGS+=(--eval-batch-size "$EVAL_BATCH_SIZE")
fi
if [[ -n "${EVAL_TENSOR_PARALLEL_SIZE:-}" ]]; then
  TRAIN_ARGS+=(--eval-tensor-parallel-size "$EVAL_TENSOR_PARALLEL_SIZE")
fi
if [[ -n "${EVAL_GPU_MEMORY_UTILIZATION:-}" ]]; then
  TRAIN_ARGS+=(--eval-gpu-memory-utilization "$EVAL_GPU_MEMORY_UTILIZATION")
fi

CMD=(python -m exp.finetune.run_axolotl "${TRAIN_ARGS[@]}")

if [[ "$NO_AUTO_EVAL" == "1" ]]; then
  AUTO_EVAL_MSG="off"
else
  AUTO_EVAL_MSG="on (default)"
fi
echo "=== Train smoke (auto-eval=$AUTO_EVAL_MSG) ==="
echo "  model=$MODEL task=$TASK optimizer=$OPTIMIZER adaptation=$ADAPTATION seed=$SEED"
echo "  num_gpus=$NUM_GPUS gpu_ids=$GPU_IDS"
echo "+ ${CMD[*]}"
exec "${CMD[@]}"
