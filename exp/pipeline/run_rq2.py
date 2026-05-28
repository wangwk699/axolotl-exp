"""End-to-end RQ2: LoRA or QLoRA -> unmerged/merged eval -> PTQ x3."""

from __future__ import annotations

import argparse
import subprocess
import sys

from exp.registry import ArtifactRegistry, RunKey


def _run(module: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", module, *args]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--num-eval", type=int, default=1000)
    parser.add_argument("--adaptation", choices=["lora", "qlora"], required=True)
    parser.add_argument("--skip-if-done", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--gpu-ids", default=None)
    args = parser.parse_args()

    key = RunKey(
        model=args.model,
        task=args.task,
        optimizer=args.optimizer,
        seed=args.seed,
        adaptation=args.adaptation,
    )
    common = [
        "--model",
        args.model,
        "--task",
        args.task,
        "--optimizer",
        args.optimizer,
        "--seed",
        str(args.seed),
        "--num-train",
        str(args.num_train),
        "--num-eval",
        str(args.num_eval),
    ]
    train_args = list(common)
    if args.skip_if_done:
        train_args.append("--skip-if-done")
    train_args.extend(["--num-gpus", str(args.num_gpus)])
    if args.gpu_ids:
        train_args.extend(["--gpu-ids", args.gpu_ids])

    _run(
        "exp.finetune.run_axolotl",
        [
            "--rq",
            "2",
            "--adaptation",
            args.adaptation,
            "--no-auto-eval",
            *train_args,
        ],
    )
    _run(
        "exp.merge.run_merge",
        [*common, "--adaptation", args.adaptation],
    )

    _run(
        "exp.eval.run_lm_eval",
        [*common, "--adaptation", args.adaptation, "--stage", "lora_unmerged"],
    )
    _run(
        "exp.eval.run_lm_eval",
        [*common, "--adaptation", args.adaptation, "--stage", "merged"],
    )

    _run(
        "exp.ptq.run_smoothquant",
        [*common, "--adaptation", args.adaptation],
    )
    for track in ("w4a16", "w4a4"):
        _run(
            "exp.ptq.run_omniquant",
            [*common, "--adaptation", args.adaptation, "--track", track],
        )

    ArtifactRegistry().write_meta(key, "rq2_pipeline_complete")
    _run("exp.analyze.aggregate_rq2", ["--append"])


if __name__ == "__main__":
    main()
