# Optimizer × Quantization Robustness Experiment

Implements [final_experiment_plan_clean_task_v1_4.md](final_experiment_plan_clean_task_v1_4.md).

## Setup

```bash
source env.sh && source .venv/bin/activate
bash scripts/setup_external.sh
python -m exp.data.prepare_all
chmod +x scripts/*.sh
```

Optional Shampoo:

```bash
pip install distributed-shampoo
```

## Single run

```bash
# RQ1
python -m exp.pipeline.run_rq1 --model llama3-8b --task gsm8k --optimizer adamw --seed 42

# RQ2 LoRA
python -m exp.pipeline.run_rq2 --model llama3-8b --task gsm8k --optimizer adamw --seed 42 --adaptation lora

# RQ2 QLoRA
python -m exp.pipeline.run_rq2 --model llama3-8b --task gsm8k --optimizer adamw --seed 42 --adaptation qlora

# RQ3
python -m exp.pipeline.run_rq3 --model llama3-8b --task gsm8k --optimizer adamw --seed 42 --adaptation full_ft
```

## Full matrix

```bash
bash scripts/run_rq1_matrix.sh --dry-run   # preview
bash scripts/run_rq1_matrix.sh
bash scripts/run_rq2_matrix.sh
bash scripts/run_rq3_matrix.sh
```

## Artifact layout

```
artifacts/{model}/{task}/{optimizer}/{seed}/{adaptation}/
  fp/ | lora_unmerged/ | merged/ | ptq/{track}/ | qat/{track}/ | meta.json
```

Results: `results/rq1_full_ft_ptq.csv`, `results/rq2_lora_ft_ptq.csv`, `results/rq3_qat.csv`.

## Tool map

| Stage | Tool |
|-------|------|
| Full-FT / LoRA / QLoRA | Axolotl (`exp/finetune`) |
| LoRA merge | Axolotl (`exp/merge`) |
| PTQ W8A8 | SmoothQuant (`exp/ptq/run_smoothquant.py`) |
| PTQ W4A16/W4A4 | OmniQuant (`exp/ptq/run_omniquant.py`) |
| QAT W4A16 Full | Axolotl (`exp/qat/run_axolotl_qat_w4a16.py`) |
| QAT W8A8/W4A4 Full | torchao fake-quant (`exp/qat/run_full_fake_quant.py`) |
| LoRA-QAT | QA-LoRA (`exp/qat/run_qalora.py`) |
| Shampoo | `exp/optimizers/shampoo_trainer.py` |
| Eval | lm-eval (`exp/eval/run_lm_eval.py`) |
