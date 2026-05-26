#!/usr/bin/env bash
# RQ1 smoke run: Full-FT -> FP eval (PTQ optional).
#
# Usage:
#   bash scripts/run_rq1_smoke.sh
#   bash scripts/run_rq1_smoke.sh 2              # 2 GPUs (default ids 0,1)
#   bash scripts/run_rq1_smoke.sh 3 0,1,2        # 3 GPUs on ids 0,1,2
#   bash scripts/run_rq1_smoke.sh 4 4,5,6,7      # 4 GPUs on ids 4,5,6,7
#
# Override defaults via env, e.g.:
#   MODEL=llama3-8b TASK=sst2 SKIP_PTQ=0 bash scripts/run_rq1_smoke.sh 2
#
# Debug RQ1 pipeline only (attach to localhost:6001):
#   DEBUG=1 bash scripts/run_rq1_smoke.sh 2
#
# Debug Axolotl train logic in-process (dataset -> model -> train):
#   bash scripts/debug_axolotl_train.sh
#   DEBUG=1 bash scripts/debug_axolotl_train.sh   # then attach "Attach Axolotl Train (6001)"
#   Or F5: "Axolotl Train (in-process)" in .vscode/launch.json
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source env.sh
source .venv/bin/activate

NUM_GPUS="${1:-${NUM_GPUS:-2}}"
GPU_IDS="${2:-${GPU_IDS:-}}"

MODEL="${MODEL:-qwen3-8b}"
TASK="${TASK:-gsm8k}"
OPTIMIZER="${OPTIMIZER:-adamw}"
SEED="${SEED:-42}"
SKIP_PTQ="${SKIP_PTQ:-1}"

if [[ -z "$GPU_IDS" ]]; then
  GPU_IDS="$(python3 - <<PY
n = int("${NUM_GPUS}")
print(",".join(str(i) for i in range(n)))
PY
)"
fi

RQ1_ARGS=(
  --model "$MODEL"
  --task "$TASK"
  --optimizer "$OPTIMIZER"
  --seed "$SEED"
  --num-gpus "$NUM_GPUS"
  --gpu-ids "$GPU_IDS"
)

if [[ "${DEBUG:-0}" == "1" ]]; then
  DEBUG_PORT="${DEBUG_PORT:-6001}"
  CMD=(
    python -m debugpy --listen "$DEBUG_PORT" --wait-for-client
    -m exp.pipeline.run_rq1
    "${RQ1_ARGS[@]}"
  )
else
  CMD=(python -m exp.pipeline.run_rq1 "${RQ1_ARGS[@]}")
fi

if [[ "$SKIP_PTQ" == "1" ]]; then
  CMD+=(--skip-ptq)
fi

echo "=== RQ1 smoke run ==="
echo "  model=$MODEL task=$TASK optimizer=$OPTIMIZER seed=$SEED"
echo "  num_gpus=$NUM_GPUS gpu_ids=$GPU_IDS skip_ptq=$SKIP_PTQ debug=${DEBUG:-0}"
echo "+ ${CMD[*]}"
exec "${CMD[@]}"
