# Experiment orchestration (v1.4)

Python package for running the optimizer × quantization robustness study.

See [final_experiment_plan_clean_task_v1_4.md](../final_experiment_plan_clean_task_v1_4.md) for the research protocol.

## Quick start

```bash
source env.sh && source .venv/bin/activate

# Prepare datasets
python -m exp.data.prepare_all

# RQ1 phase A then B
python -m exp.pipeline.run_rq1_train --model qwen3-8b --task metamath --optimizer adamw --seed 42
python -m exp.pipeline.run_rq1_ptq --model qwen3-8b --task metamath --optimizer adamw --seed 42

# Full matrix (dry-run first)
bash scripts/run_rq1_train_matrix.sh --dry-run
bash scripts/run_rq1_ptq_matrix.sh --dry-run
```

## Layout

- `registry.py` — RunKey → artifact paths
- `render_config.py` — Axolotl YAML generation
- `pipeline/` — end-to-end RQ1/RQ2/RQ3 runners
- `ptq/` — SmoothQuant / OmniQuant wrappers
- `qat/` — Axolotl QAT, fake-quant QAT, QA-LoRA wrappers
- `eval/` — unified lm-eval
- `analyze/` — CSV aggregation
