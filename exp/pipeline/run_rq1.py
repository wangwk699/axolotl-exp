"""End-to-end RQ1: Full-FT -> FP eval -> PTQ x3 -> quant eval."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from exp.registry import ArtifactRegistry, RunKey, get_project_root


def _run_module(module: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", module, *args]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-if-done", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-ptq", action="store_true")
    args = parser.parse_args()

    key = RunKey(
        model=args.model,
        task=args.task,
        optimizer=args.optimizer,
        seed=args.seed,
        adaptation="full_ft",
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

    if not args.skip_train:
        _run_module(
            "exp.finetune.run_axolotl",
            ["--rq", "1", "--adaptation", "full_ft", *common],
        )

    if args.task != "humaneval":
        _run_module(
            "exp.eval.run_lm_eval",
            [
                *common,
                "--adaptation",
                "full_ft",
                "--stage",
                "fp",
            ],
        )

    if not args.skip_ptq:
        _run_module(
            "exp.ptq.run_smoothquant",
            [*common, "--adaptation", "full_ft"],
        )
        for track in ("w4a16", "w4a4"):
            _run_module(
                "exp.ptq.run_omniquant",
                [*common, "--adaptation", "full_ft", "--track", track],
            )

    reg = ArtifactRegistry()
    reg.write_meta(key, "rq1_pipeline_complete")
    _run_module("exp.analyze.aggregate_rq1", ["--append"])


if __name__ == "__main__":
    main()
