#!/usr/bin/env bash
# Debug Axolotl train in-process (load data -> model -> train).
#
# Single-process path: calls axolotl.cli.train.do_cli directly (no subprocess).
# Default: 1 GPU so you can step through without torchrun/FSDP workers.
#
# Usage:
#   bash scripts/debug_axolotl_train.sh
#   bash scripts/debug_axolotl_train.sh 0          # use GPU 0 only
#   MODEL=llama3-8b TASK=sst2 bash scripts/debug_axolotl_train.sh
#   MODEL=qwen3-1.7b TASK=sst2 GPU_IDS=0 DEBUG=1 bash scripts/debug_axolotl_train.sh
#
# Attach (terminal + VS Code / Cursor):
#   DEBUG=1 bash scripts/debug_axolotl_train.sh
#   Then: Run and Debug -> "Attach Axolotl Train (6001)"
#
# Or use VS Code launch: "Axolotl Train (in-process)" (F5, no attach needed).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source env.sh
source .venv/bin/activate

GPU_IDS="${1:-${GPU_IDS:-0}}"
MODEL="${MODEL:-qwen3-8b}"
TASK="${TASK:-gsm8k}"
OPTIMIZER="${OPTIMIZER:-adamw}"
SEED="${SEED:-42}"
RQ="${RQ:-1}"
ADAPTATION="${ADAPTATION:-full_ft}"
NUM_GPUS="${NUM_GPUS:-1}"

ARGS=(
  --rq "$RQ"
  --adaptation "$ADAPTATION"
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
    python -m exp.finetune.debug_axolotl_train
    "${ARGS[@]}"
    --attach-port "$DEBUG_PORT"
  )
else
  CMD=(python -m exp.finetune.debug_axolotl_train "${ARGS[@]}")
fi

echo "=== Debug Axolotl train (in-process) ==="
echo "  model=$MODEL task=$TASK optimizer=$OPTIMIZER adaptation=$ADAPTATION"
echo "  num_gpus=$NUM_GPUS gpu_ids=$GPU_IDS debug=${DEBUG:-0}"
echo "+ ${CMD[*]}"
exec "${CMD[@]}"
