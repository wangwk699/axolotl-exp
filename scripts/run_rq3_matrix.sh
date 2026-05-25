#!/usr/bin/env bash
# Run full RQ3 matrix: full_ft + lora + qlora QAT paths.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source env.sh
source .venv/bin/activate

DRY_RUN="${1:-}"
MODELS=(llama3-8b qwen3-8b)
TASKS=(sst2 rte boolq multirc gsm8k humaneval commonsenseqa)
OPTS=(adamw muon shampoo)
SEEDS=(42 43 44)
ADAPTS=(full_ft lora qlora)

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    for opt in "${OPTS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        for adapt in "${ADAPTS[@]}"; do
          CMD=(python -m exp.pipeline.run_rq3 --model "$model" --task "$task" --optimizer "$opt" --seed "$seed" --adaptation "$adapt" --skip-if-done)
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

python -m exp.analyze.aggregate_rq3
