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
    parser.add_argument("--adaptation", choices=["lora", "qlora"], required=True)
    parser.add_argument("--skip-if-done", action="store_true")
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
    ]
    if args.skip_if_done:
        common.append("--skip-if-done")

    _run(
        "exp.finetune.run_axolotl",
        ["--rq", "2", "--adaptation", args.adaptation, *common],
    )
    _run(
        "exp.merge.run_merge",
        [*common, "--adaptation", args.adaptation],
    )

    if args.task != "humaneval":
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
