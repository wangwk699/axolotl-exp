#!/usr/bin/env bash
# Run full RQ2 matrix: both lora and qlora adaptations.
#
# Train tasks: sst2 rte boolq metamath codefeedback
# Eval via tasks.yaml eval_task (e.g. metamath -> gsm8k).
#
# Prepare data: python -m exp.data.prepare_all
# Dry-run: bash scripts/run_rq2_matrix.sh --dry-run
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source env.sh
source .venv/bin/activate

DRY_RUN="${1:-}"
MODELS=(llama3-8b qwen3-8b)
TASKS=(sst2 rte boolq metamath codefeedback)
OPTS=(adamw muon shampoo)
SEEDS=(42 43 44)
ADAPTS=(lora qlora)
NUM_TRAIN="${NUM_TRAIN:-1000}"
NUM_EVAL="${NUM_EVAL:-1000}"

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    for opt in "${OPTS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        for adapt in "${ADAPTS[@]}"; do
          CMD=(python -m exp.pipeline.run_rq2 --model "$model" --task "$task" --optimizer "$opt" --seed "$seed" --num-train "$NUM_TRAIN" --num-eval "$NUM_EVAL" --adaptation "$adapt" --skip-if-done)
          if [[ "$DRY_RUN" == "--dry-run" ]]; then
            echo "${CMD[*]}"
          else
            PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "${CMD[@]}"
          fi
        done
      done
    done
  done
done

python -m exp.analyze.aggregate_rq2
