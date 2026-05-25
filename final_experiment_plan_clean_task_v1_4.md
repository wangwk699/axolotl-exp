# Experiment Plan: Optimizer Effects on Quantization Robustness

Version: 1.4  
Scope: clean task description  
Organization: three chapters aligned with RQ1, RQ2, and RQ3

---

## 0. Shared Setup

### Optimizers

```text
AdamW, Muon, Shampoo
```

### Quantization Tracks

```text
W8A8, W4A16, W4A4
```

Suggested method mapping:

| Track | PTQ method | QAT method |
|---|---|---|
| W8A8 | SmoothQuant | fake-quant QAT |
| W4A16 | OmniQuant | weight-only fake-quant QAT |
| W4A4 | OmniQuant | low-bit fake-quant QAT |

### Task Suite

#### Natural Language Understanding

```text
SST-2, RTE, BoolQ, MultiRC
```

#### Natural Language Generation / Reasoning

```text
Math: GSM8K
Code: HumanEval
Commonsense reasoning: CommonsenseQA
```

### Metrics

Use the downstream metric of each task.

| Task | Primary metric |
|---|---|
| SST-2 | accuracy |
| RTE | accuracy |
| BoolQ | accuracy |
| MultiRC | official F1 / EM |
| GSM8K | final-answer exact match |
| HumanEval | pass@1 |
| CommonsenseQA | final-answer accuracy |

### Common Result Fields

For each quantized run, report:

```text
score_fp
score_quant
delta_quant = score_fp - score_quant
retention = score_quant / score_fp
```

---

## 1. Chapter 1 / RQ1: Full-FT under PTQ

### Research Question

Under full-parameter fine-tuning, does optimizer choice affect PTQ robustness?

### Experiment Scope

```text
Adaptation: Full-FT
Quantization: PTQ
Tracks: W8A8, W4A16, W4A4
Optimizers: AdamW, Muon, Shampoo
Tasks: NLU + NLG/reasoning task suite
```

### Experiment Flow

For each model, task, optimizer, and seed:

```text
pretrained model
-> full-parameter fine-tuning
-> full-precision evaluation
-> PTQ W8A8 evaluation
-> PTQ W4A16 evaluation
-> PTQ W4A4 evaluation
```

### Main Comparison

Compare optimizers by:

```text
score_ptq
delta_ptq
retention_ptq
```

### Output Table

```text
results/rq1_full_ft_ptq.csv
```

Required columns:

```text
model, task, optimizer, seed,
adaptation_method, quantization_mode, bitwidth_track,
score_fp, score_quant, delta_quant, retention
```

---

## 2. Chapter 2 / RQ2: LoRA-FT under PTQ

### Research Question

Under LoRA fine-tuning, does optimizer choice affect PTQ robustness, and how does this differ from Full-FT?

### Experiment Scope

```text
Adaptation: LoRA-FT
Quantization: PTQ
Tracks: W8A8, W4A16, W4A4
Optimizers: AdamW, Muon, Shampoo
Tasks: NLU + NLG/reasoning task suite
```

### Experiment Flow

For each model, task, optimizer, and seed:

```text
pretrained model
-> LoRA/QLoRA fine-tuning  
-> evaluate LoRA full-precision model
-> merge LoRA into base model
-> evaluate merged full-precision model
-> PTQ W8A8 evaluation
-> PTQ W4A16 evaluation
-> PTQ W4A4 evaluation
```

### Main Comparison

Compare optimizers by:

```text
score_ptq
delta_ptq
retention_ptq
```

Compare LoRA-FT against Chapter 1 Full-FT by:

```text
delta_ptq_lora vs delta_ptq_full
retention_ptq_lora vs retention_ptq_full
```

### Output Table

```text
results/rq2_lora_ft_ptq.csv
```

Required columns:

```text
model, task, optimizer, seed,
adaptation_method, quantization_mode, bitwidth_track,
score_fp, score_quant, delta_quant, retention,
score_lora_unmerged, score_lora_merged
```

---

## 3. Chapter 3 / RQ3: Full-FT and LoRA-FT under QAT

### Research Question

Are optimizer rankings under QAT consistent with optimizer rankings under PTQ?

### Experiment Scope

```text
Adaptation: Full-FT and LoRA-FT
Quantization: Direct QAT
Tracks: W8A8, W4A16, W4A4
Optimizers: AdamW, Muon, Shampoo
Tasks: NLU + NLG/reasoning task suite
```

### Full-QAT Flow

For each model, task, optimizer, bitwidth track, and seed:

```text
pretrained model
-> prepare fake-quant model
-> full-parameter QAT fine-tuning
-> quantized evaluation
```

### LoRA-QAT Flow—QA-LoRA

For each model, task, optimizer, bitwidth track, and seed:

```text
pretrained model
-> insert LoRA adapters
-> prepare fake-quant LoRA model
-> LoRA QAT fine-tuning
-> merge or integrate LoRA for quantized evaluation
-> quantized evaluation
```

### Main Comparison

For each task, adaptation method, and bitwidth track:

```text
rank_ptq(AdamW, Muon, Shampoo)
rank_qat(AdamW, Muon, Shampoo)
```

Report:

```text
score_qat
delta_qat
retention_qat
Kendall tau between PTQ rank and QAT rank
Spearman rho between PTQ rank and QAT rank
```

### Output Tables

```text
results/rq3_qat.csv
results/rq3_ptq_qat_rank_correlation.csv
```

Required columns for `rq3_qat.csv`:

```text
model, task, optimizer, seed,
adaptation_method, quantization_mode, bitwidth_track,
score_fp, score_quant, delta_quant, retention
```

Required columns for `rq3_ptq_qat_rank_correlation.csv`:

```text
model, task, adaptation_method, bitwidth_track,
rank_ptq, rank_qat, kendall_tau, spearman_rho
```

---

## 4. Final Chapter Map

| Chapter | RQ | Adaptation | Quantization | Tracks | Main purpose |
|---|---|---|---|---|---|
| Chapter 1 | RQ1 | Full-FT | PTQ | W8A8, W4A16, W4A4 | optimizer effect under Full-FT PTQ |
| Chapter 2 | RQ2 | LoRA-FT | PTQ | W8A8, W4A16, W4A4 | optimizer effect under LoRA-FT PTQ |
| Chapter 3 | RQ3 | Full-FT + LoRA-FT | QAT | W8A8, W4A16, W4A4 | PTQ vs QAT optimizer ranking consistency |
