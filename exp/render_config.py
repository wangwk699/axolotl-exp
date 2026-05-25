"""Generate Axolotl YAML configs from templates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from exp.registry import ArtifactRegistry, RunKey, get_project_root, load_yaml_config


def _optimizer_block(optimizer: str) -> dict[str, Any]:
    opt_cfg = load_yaml_config("optimizers.yaml")["optimizers"][optimizer]
    block: dict[str, Any] = {
        "optimizer": opt_cfg["axolotl_optimizer"],
        "learning_rate": opt_cfg["learning_rate"],
    }
    if opt_cfg.get("use_fsdp2"):
        block["fsdp_version"] = opt_cfg.get("fsdp_version", 2)
        block["fsdp_config"] = {
            "offload_params": False,
            "cpu_ram_efficient_loading": True,
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "state_dict_type": "FULL_STATE_DICT",
            "sharding_strategy": "FULL_SHARD",
            "reshard_after_forward": True,
            "activation_checkpointing": True,
        }
    return block


def _base_training_config(key: RunKey, rq: int, adaptation: str) -> dict[str, Any]:
    models = load_yaml_config("models.yaml")
    defaults = models["defaults"]
    model_cfg = models["models"][key.model]
    reg = ArtifactRegistry()
    reg.ensure_dirs(key)

    data_path = get_project_root() / "data" / "processed" / key.task / "train.jsonl"
    if key.task != "humaneval" and not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Run: python -m exp.data.prepare_all"
        )

    cfg: dict[str, Any] = {
        "base_model": model_cfg["hf_id"],
        "strict": False,
        "datasets": [
            {
                "path": str(data_path),
                "type": "alpaca",
                "ds_type": "json",
            }
        ],
        "val_set_size": defaults["val_set_size"],
        "sequence_len": defaults["sequence_len"],
        "sample_packing": True,
        "eval_sample_packing": True,
        "micro_batch_size": defaults["micro_batch_size"],
        "gradient_accumulation_steps": defaults["gradient_accumulation_steps"],
        "num_epochs": defaults["num_epochs"],
        "lr_scheduler": "cosine",
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": 1,
        "warmup_ratio": defaults["warmup_ratio"],
        "evals_per_epoch": 2,
        "saves_per_epoch": 1,
        "weight_decay": 0.0,
        "attn_implementation": "flash_attention_2",
        "seed": key.seed,
    }

    if model_cfg.get("chat_template"):
        cfg["chat_template"] = model_cfg["chat_template"]
    if model_cfg.get("pad_token"):
        cfg["special_tokens"] = {"pad_token": model_cfg["pad_token"]}

    cfg.update(_optimizer_block(key.optimizer))

    if adaptation == "full_ft":
        cfg["output_dir"] = str(reg.fp_dir(key))
    elif adaptation in ("lora", "qlora"):
        cfg["adapter"] = adaptation
        cfg["output_dir"] = str(reg.lora_unmerged_dir(key))
        cfg["lora_r"] = defaults["lora_r"]
        cfg["lora_alpha"] = defaults["lora_alpha"]
        cfg["lora_dropout"] = defaults["lora_dropout"]
        cfg["lora_target_linear"] = True
        if adaptation == "qlora":
            cfg["load_in_4bit"] = True
            cfg["load_in_8bit"] = False
            cfg["bnb_4bit_quant_type"] = "nf4"
            cfg["bnb_4bit_use_double_quant"] = True
    elif adaptation == "full_qat_w4a16":
        cfg["output_dir"] = str(reg.qat_dir(key, "w4a16"))
        cfg["qat"] = {
            "weight_dtype": "int4",
            "activation_dtype": None,
            "group_size": 32,
            "quantize_embedding": False,
        }
        cfg["fsdp_version"] = 2
    else:
        raise ValueError(f"Unknown adaptation {adaptation}")

    if key.model == "qwen3-8b" and adaptation in ("lora", "qlora", "full_ft"):
        cfg.setdefault("fsdp_config", {})
        cfg["fsdp_config"]["transformer_layer_cls_to_wrap"] = "Qwen3DecoderLayer"

    task_cfg = load_yaml_config("tasks.yaml")["tasks"][key.task]
    cfg["lm_eval_tasks"] = [task_cfg["lm_eval_task"]]
    cfg["lm_eval_model"] = cfg["output_dir"]

    return cfg


def render_config(key: RunKey, rq: int, adaptation: str) -> Path:
    cfg = _base_training_config(key, rq, adaptation)
    sub = {
        "full_ft": f"rq{rq}/full_ft",
        "lora": f"rq{rq}/lora",
        "qlora": f"rq{rq}/qlora",
        "full_qat_w4a16": f"rq{rq}/full_qat_w4a16",
    }[adaptation]
    out_dir = get_project_root() / "configs" / "axolotl" / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{key.slug()}.yml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return out_path


def render_merge_config(key: RunKey) -> Path:
    models = load_yaml_config("models.yaml")
    reg = ArtifactRegistry()
    cfg = {
        "base_model": models["models"][key.model]["hf_id"],
        "adapter": key.adaptation,
        "lora_model_dir": str(reg.lora_unmerged_dir(key)),
        "output_dir": str(reg.merged_dir(key)),
        "lm_eval_tasks": [load_yaml_config("tasks.yaml")["tasks"][key.task]["lm_eval_task"]],
        "lm_eval_model": str(reg.merged_dir(key)),
    }
    out_dir = get_project_root() / "configs" / "axolotl" / "merge"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{key.slug()}.yml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path
