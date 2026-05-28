"""Run Axolotl fine-tuning with optional Shampoo hook."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from exp.finetune.axolotl_launch import build_train_command, set_visible_gpus
from exp.registry import ArtifactRegistry, RunKey, get_project_root, load_yaml_config
from exp.render_config import render_config


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env or os.environ.copy())

def _auto_eval(
    *,
    key: RunKey,
    eval_task: str,
    stage: str,
    backend: str | None,
    batch_size: str | None,
    tensor_parallel_size: int | None,
    gpu_memory_utilization: float | None,
    gpu_ids: str | None,
) -> None:
    from exp.eval.run_lm_eval import run_lm_eval

    reg = ArtifactRegistry()
    if stage == "fp":
        model_path = reg.fp_dir(key) if key.adaptation == "full_ft" else reg.merged_dir(key)
    elif stage == "lora_unmerged":
        model_path = reg.lora_unmerged_dir(key)
    elif stage == "merged":
        model_path = reg.merged_dir(key)
    else:
        raise ValueError(f"Unsupported stage for auto-eval: {stage}")

    eval_out = model_path / "lm_eval_out" / eval_task
    metrics = run_lm_eval(
        model_path,
        eval_task,
        eval_out,
        backend=backend,
        batch_size=batch_size,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        gpu_ids=gpu_ids,
    )
    out_metrics = reg.run_dir(key) / f"metrics_auto_eval_{eval_task}.json"
    out_metrics.write_text(
        __import__("json").dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    reg.write_meta(
        key,
        f"auto_eval_{stage}",
        metrics={"task": eval_task, "score": metrics["score"], "metric": metrics["metric"]},
    )
    print(f"auto-eval {eval_task}: score={metrics['score']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rq", type=int, required=True)
    parser.add_argument("--adaptation", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-train",
        type=int,
        default=1000,
        help="Number of training samples to use (sampled randomly from canonical train.jsonl).",
    )
    parser.add_argument(
        "--num-eval",
        type=int,
        default=1000,
        help="Number of eval samples to use (sampled randomly from canonical eval.jsonl).",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated GPU ids (e.g. 0,1). Defaults to 0..N-1 when unset.",
    )
    parser.add_argument("--skip-if-done", action="store_true")
    parser.add_argument(
        "--auto-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run lm-eval automatically after training (default: true).",
    )
    parser.add_argument("--eval-backend", choices=["vllm", "hf"], default=None)
    parser.add_argument("--eval-batch-size", default=None, help="e.g. auto, 32")
    parser.add_argument("--eval-tensor-parallel-size", type=int, default=None)
    parser.add_argument("--eval-gpu-memory-utilization", type=float, default=None)
    args = parser.parse_args()

    if args.num_gpus < 1:
        raise ValueError("--num-gpus must be >= 1")

    key = RunKey(
        model=args.model,
        task=args.task,
        optimizer=args.optimizer,
        seed=args.seed,
        adaptation=args.adaptation,
    )
    reg = ArtifactRegistry()
    stage = f"finetune_{args.adaptation}"
    if args.skip_if_done and reg.stage_done(key, stage):
        print(f"Skip {stage} (already done)")
        return

    set_visible_gpus(args.gpu_ids, args.num_gpus)

    # Ensure canonical data exists on disk.
    from exp.data.prepare_all import prepare_task

    prepare_task(args.task, seed=args.seed)

    # Materialize per-run subset datasets under artifacts (seed fixed to 42).
    train_data_path: Path | None = None
    eval_data_path: Path | None = None
    if args.num_train > 0 or args.num_eval > 0:
        from exp.data.subset_jsonl import sample_jsonl

        canonical_root = get_project_root() / "data" / "processed" / args.task
        canonical_train = canonical_root / "train.jsonl"
        canonical_eval = canonical_root / "eval.jsonl"
        ds_dir = reg.run_dir(key) / "datasets"
        if args.num_train > 0:
            train_data_path = sample_jsonl(
                canonical_train,
                ds_dir / f"train_n{args.num_train}_seed42.jsonl",
                num_samples=args.num_train,
                seed=42,
            )
        if args.num_eval > 0:
            eval_data_path = sample_jsonl(
                canonical_eval,
                ds_dir / f"eval_n{args.num_eval}_seed42.jsonl",
                num_samples=args.num_eval,
                seed=42,
            )

    cfg_path = render_config(
        key,
        args.rq,
        args.adaptation,
        num_gpus=args.num_gpus,
        train_data_path=train_data_path,
        eval_data_path=eval_data_path,
    )
    env = os.environ.copy()
    if args.optimizer == "shampoo":
        env["EXP_USE_SHAMPOO"] = "1"
        root = str(get_project_root())
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")

    cmd = build_train_command(
        cfg_path,
        num_gpus=args.num_gpus,
        optimizer=args.optimizer,
    )
    _run(cmd, env=env)
    reg.write_meta(
        key,
        stage,
        axolotl_config=cfg_path,
        command=" ".join(cmd),
        extra={"num_gpus": args.num_gpus, "gpu_ids": args.gpu_ids or env.get("CUDA_VISIBLE_DEVICES")},
    )

    if args.auto_eval:
        task_cfg = load_yaml_config("tasks.yaml")["tasks"].get(args.task, {})
        eval_task = task_cfg.get("eval_task", args.task)
        # After training we only have fp or lora_unmerged artifacts here.
        eval_stage = "fp" if args.adaptation == "full_ft" else "lora_unmerged"
        _auto_eval(
            key=key,
            eval_task=eval_task,
            stage=eval_stage,
            backend=args.eval_backend,
            batch_size=args.eval_batch_size,
            tensor_parallel_size=args.eval_tensor_parallel_size,
            gpu_memory_utilization=args.eval_gpu_memory_utilization,
            gpu_ids=args.gpu_ids,
        )


if __name__ == "__main__":
    main()
