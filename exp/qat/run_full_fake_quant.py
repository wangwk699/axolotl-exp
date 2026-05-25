"""Full-parameter fake-quant QAT for W8A8 and W4A4 (torchao-based)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from exp.registry import ArtifactRegistry, RunKey, get_project_root, load_yaml_config


def _load_sft_batches(task: str, tokenizer, max_len: int, batch_size: int):
    data_path = get_project_root() / "data" / "processed" / task / "train.jsonl"
    texts = []
    with data_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(
                f"{row['instruction']}\n{row.get('input', '')}\n{row['output']}"
            )

    def collate(batch):
        return tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )

    return DataLoader(texts, batch_size=batch_size, shuffle=True, collate_fn=collate)


def _prepare_qat(model, track: str):
    from torchao.quantization import quantize_
    from torchao.quantization.qat import QATConfig
    from torchao.quantization.qat.fake_quantize_config import (
        Float8FakeQuantizeConfig,
        Int4WeightFakeQuantizeConfig,
        IntxFakeQuantizeConfig,
    )

    if track == "w8a8":
        weight_fq = IntxFakeQuantizeConfig(
            dtype=torch.int8, granularity="per_channel", is_symmetric=False
        )
        act_fq = IntxFakeQuantizeConfig(
            dtype=torch.int8, granularity="per_token", is_symmetric=False
        )
        quantize_(model, QATConfig(weight_config=weight_fq, activation_config=act_fq))
    elif track == "w4a4":
        weight_fq = Int4WeightFakeQuantizeConfig(group_size=32)
        act_fq = IntxFakeQuantizeConfig(
            dtype=torch.int8, granularity="per_token", is_symmetric=False
        )
        quantize_(model, QATConfig(weight_config=weight_fq, activation_config=act_fq))
    else:
        raise ValueError(track)


def train_qat(key: RunKey, track: str, steps: int = 200, lr: float = 2e-5) -> Path:
    models = load_yaml_config("models.yaml")
    reg = ArtifactRegistry()
    out_dir = reg.qat_dir(key, track)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_id = models["models"][key.model]["hf_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    _prepare_qat(model, track)
    model.train()

    loader = _load_sft_batches(key.task, tokenizer, max_len=512, batch_size=1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = get_cosine_schedule_with_warmup(opt, 10, steps)

    step = 0
    while step < steps:
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            out = model(**batch, labels=labels)
            loss = out.loss
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step >= steps:
                break

    from axolotl.utils.quantization import convert_qat_model

    convert_qat_model(model)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--track", choices=["w8a8", "w4a4"], required=True)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    key = RunKey(
        model=args.model,
        task=args.task,
        optimizer=args.optimizer,
        seed=args.seed,
        adaptation="full_ft",
    )
    reg = ArtifactRegistry()
    stage = f"qat_{args.track}"
    if reg.stage_done(key, stage):
        print("Skip (done)")
        return

    out = train_qat(key, args.track, steps=args.steps)
    reg.write_meta(key, stage, extra={"output_dir": str(out)})


if __name__ == "__main__":
    main()
