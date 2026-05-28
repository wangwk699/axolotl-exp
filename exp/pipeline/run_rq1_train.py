"""RQ1 phase A: Full-FT train -> FP lm-eval -> aggregate FP scores."""

from __future__ import annotations

import argparse

from exp.pipeline._common import run_module
from exp.registry import ArtifactRegistry, RunKey


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RQ1 train phase: fine-tune, FP evaluation, aggregate FP metrics.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--num-eval", type=int, default=1000)
    parser.add_argument("--skip-if-done", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated GPU ids (e.g. 0,1). Defaults to 0..num-gpus-1.",
    )
    parser.add_argument("--eval-backend", choices=["vllm", "hf"], default=None)
    parser.add_argument("--eval-batch-size", default=None)
    parser.add_argument("--eval-gpu-ids", default=None, help="GPUs for lm-eval (defaults to --gpu-ids)")
    args = parser.parse_args()

    key = RunKey(
        model=args.model,
        task=args.task,
        optimizer=args.optimizer,
        seed=args.seed,
        adaptation="full_ft",
    )
    # Args shared across train/eval sub-commands.
    common = [
        "--model",
        args.model,
        "--task",
        args.task,
        "--optimizer",
        args.optimizer,
        "--seed",
        str(args.seed),
    ]
    # Training supports controlling dataset subsampling; lm-eval does not.
    train_args = [
        *common,
        "--num-train",
        str(args.num_train),
        "--num-eval",
        str(args.num_eval),
    ]
    if args.skip_if_done:
        train_args.append("--skip-if-done")
    train_args.extend(["--num-gpus", str(args.num_gpus)])
    if args.gpu_ids:
        train_args.extend(["--gpu-ids", args.gpu_ids])

    if not args.skip_train:
        run_module(
            "exp.finetune.run_axolotl",
            [
                "--rq",
                "1",
                "--adaptation",
                "full_ft",
                "--no-auto-eval",
                *train_args,
            ],
        )

    if not args.skip_eval:
        eval_args = [
            *common,
            "--adaptation",
            "full_ft",
            "--stage",
            "fp",
        ]
        eval_gpu = args.eval_gpu_ids or args.gpu_ids
        if eval_gpu:
            eval_args.extend(["--gpu-ids", eval_gpu])
        if args.eval_backend:
            eval_args.extend(["--backend", args.eval_backend])
        if args.eval_batch_size:
            eval_args.extend(["--batch-size", args.eval_batch_size])
        run_module("exp.eval.run_lm_eval", eval_args)

    reg = ArtifactRegistry()
    reg.write_meta(key, "rq1_train_complete")

    if not args.skip_aggregate:
        run_module("exp.analyze.aggregate_rq1_train", [])


if __name__ == "__main__":
    main()
