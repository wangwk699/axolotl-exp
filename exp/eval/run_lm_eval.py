"""Unified lm-eval evaluation for FP / PTQ / QAT checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from exp.registry import ArtifactRegistry, RunKey, load_yaml_config


def _vllm_available() -> bool:
    if importlib.util.find_spec("vllm") is None:
        return False
    try:
        import vllm  # noqa: F401
        from vllm import LLM  # noqa: F401
    except ImportError as exc:
        print(f"vLLM import failed ({exc}); falling back to hf backend.")
        return False
    return True


def _primary_metric(task: str, results: dict[str, Any]) -> float:
    task_cfg = load_yaml_config("tasks.yaml")["tasks"][task]
    metric = task_cfg["primary_metric"]
    lm_task = task_cfg["lm_eval_task"].replace("/", ",")

    res_block = results.get("results", results)
    for key, vals in res_block.items():
        if lm_task.split("/")[-1] in key or task in key:
            if metric in vals:
                return float(vals[metric])
            if metric == "acc" and "acc,none" in vals:
                return float(vals["acc,none"])
            if metric == "exact_match" and "exact_match,strict-match" in vals:
                return float(vals["exact_match,strict-match"])
            if metric == "pass@1" and "pass@1,create_test" in vals:
                return float(vals["pass@1,create_test"])
            if metric == "f1" and "f1,macro" in vals:
                return float(vals["f1,macro"])
    raise KeyError(f"Metric {metric} not found in {json.dumps(results)[:500]}")


def _resolve_backend(requested: str | None) -> str:
    eval_cfg = load_yaml_config("eval.yaml")
    backend = requested or eval_cfg.get("backend", "vllm")
    if backend == "vllm" and not _vllm_available():
        print("vLLM not installed; falling back to hf backend. Install with: uv pip install vllm")
        return "hf"
    return backend


def _build_lm_eval_cmd(
    model_path: Path,
    task: str,
    output_path: Path,
    *,
    backend: str | None = None,
    batch_size: str | None = None,
    tensor_parallel_size: int | None = None,
    gpu_memory_utilization: float | None = None,
    gpu_ids: str | None = None,
) -> list[str]:
    eval_cfg = load_yaml_config("eval.yaml")
    backend = _resolve_backend(backend)
    task_cfg = load_yaml_config("tasks.yaml")["tasks"][task]
    lm_task = task_cfg["lm_eval_task"]

    if backend == "vllm":
        vcfg = eval_cfg["vllm"]
        tp = tensor_parallel_size or vcfg.get("tensor_parallel_size", 1)
        bs = batch_size or str(vcfg.get("batch_size", "auto"))
        mem_util = gpu_memory_utilization if gpu_memory_utilization is not None else vcfg.get("gpu_memory_utilization", 0.75)
        model_args = (
            f"pretrained={model_path},"
            f"dtype={vcfg.get('dtype', 'bfloat16')},"
            f"gpu_memory_utilization={mem_util},"
            f"max_model_len={vcfg.get('max_model_len', 4096)},"
            f"max_gen_toks={vcfg.get('max_gen_toks', 512)},"
            f"tensor_parallel_size={tp},"
            f"trust_remote_code={str(vcfg.get('trust_remote_code', True)).lower()}"
        )
        max_bs = vcfg.get("max_batch_size")
        if max_bs is not None:
            model_args += f",max_batch_size={max_bs}"
    else:
        hcfg = eval_cfg["hf"]
        bs = batch_size or str(hcfg.get("batch_size", 32))
        model_args = (
            f"pretrained={model_path},"
            f"dtype={hcfg.get('dtype', 'bfloat16')},"
            f"attn_implementation={hcfg.get('attn_implementation', 'flash_attention_2')}"
        )

    cmd = [
        "lm_eval",
        "run",
        "--model",
        backend,
        "--model_args",
        model_args,
        "--tasks",
        lm_task,
        "--batch_size",
        bs,
        "--output_path",
        str(output_path),
    ]
    num_fewshot = task_cfg.get("num_fewshot")
    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])
    return cmd


def run_lm_eval(
    model_path: Path,
    task: str,
    output_path: Path,
    *,
    backend: str | None = None,
    batch_size: str | None = None,
    tensor_parallel_size: int | None = None,
    gpu_memory_utilization: float | None = None,
    gpu_ids: str | None = None,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _build_lm_eval_cmd(
        model_path,
        task,
        output_path,
        backend=backend,
        batch_size=batch_size,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        gpu_ids=gpu_ids,
    )
    env = os.environ.copy()
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    eval_cfg = load_yaml_config("eval.yaml")
    if eval_cfg.get("hf_offline", True):
        env["HF_HUB_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
    print("+", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True, env=env)

    result_files = sorted(output_path.glob("**/*.json"))
    if not result_files:
        raise FileNotFoundError(f"No lm-eval output under {output_path}")
    data = json.loads(result_files[-1].read_text(encoding="utf-8"))
    score = _primary_metric(task, data)
    return {"raw": data, "score": score, "metric": load_yaml_config("tasks.yaml")["tasks"][task]["primary_metric"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adaptation", required=True)
    parser.add_argument(
        "--stage",
        choices=["fp", "lora_unmerged", "merged", "ptq", "qat"],
        required=True,
    )
    parser.add_argument("--track", choices=["w8a8", "w4a16", "w4a4"], default=None)
    parser.add_argument("--backend", choices=["vllm", "hf"], default=None)
    parser.add_argument("--batch-size", default=None, help="e.g. auto, 32")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--gpu-ids", default=None, help="CUDA_VISIBLE_DEVICES for eval")
    args = parser.parse_args()

    key = RunKey(
        model=args.model,
        task=args.task,
        optimizer=args.optimizer,
        seed=args.seed,
        adaptation=args.adaptation,
    )
    reg = ArtifactRegistry()

    if args.stage == "fp":
        model_path = reg.fp_dir(key) if key.adaptation == "full_ft" else reg.merged_dir(key)
        out_metrics = reg.fp_eval_metrics_path(key)
    elif args.stage == "lora_unmerged":
        model_path = reg.lora_unmerged_dir(key)
        out_metrics = reg.run_dir(key) / "metrics_lora_unmerged.json"
    elif args.stage == "merged":
        model_path = reg.merged_dir(key)
        out_metrics = reg.run_dir(key) / "metrics_lora_merged.json"
    elif args.stage == "ptq":
        if not args.track:
            raise ValueError("--track required for ptq eval")
        model_path = reg.ptq_dir(key, args.track)
        out_metrics = reg.ptq_eval_metrics_path(key, args.track)
    elif args.stage == "qat":
        if not args.track:
            raise ValueError("--track required for qat eval")
        model_path = reg.qat_dir(key, args.track)
        out_metrics = reg.qat_eval_metrics_path(key, args.track)
    else:
        raise ValueError(args.stage)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path missing: {model_path}")

    eval_out = model_path / "lm_eval_out"
    metrics = run_lm_eval(
        model_path,
        args.task,
        eval_out,
        backend=args.backend,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        gpu_ids=args.gpu_ids,
    )
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    reg.write_meta(key, f"eval_{args.stage}", metrics={"score": metrics["score"]})
    print(f"score={metrics['score']}")


if __name__ == "__main__":
    main()
