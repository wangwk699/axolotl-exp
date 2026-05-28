#!/usr/bin/env bash
# Run RQ1 phase A matrix: train + FP eval + aggregate (all train tasks).
#
# Dry-run: bash scripts/run_rq1_train_matrix.sh --dry-run
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
NUM_TRAIN="${NUM_TRAIN:-1000}"
NUM_EVAL="${NUM_EVAL:-1000}"

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    for opt in "${OPTS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        CMD=(python -m exp.pipeline.run_rq1_train --model "$model" --task "$task" --optimizer "$opt" --seed "$seed" --num-train "$NUM_TRAIN" --num-eval "$NUM_EVAL" --skip-if-done)
        if [[ "$DRY_RUN" == "--dry-run" ]]; then
          echo "${CMD[*]}"
        else
          PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "${CMD[@]}"
        fi
      done
    done
  done
done

python -m exp.analyze.aggregate_rq1_train
