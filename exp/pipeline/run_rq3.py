"""End-to-end RQ3: Full-QAT + LoRA-QAT across bitwidth tracks."""

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
    parser.add_argument("--adaptation", choices=["full_ft", "lora", "qlora"], required=True)
    parser.add_argument("--skip-if-done", action="store_true")
    args = parser.parse_args()

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

    if args.adaptation == "full_ft":
        key = RunKey(
            model=args.model,
            task=args.task,
            optimizer=args.optimizer,
            seed=args.seed,
            adaptation="full_ft",
        )
        _run("exp.qat.run_axolotl_qat_w4a16", common)
        for track in ("w8a8", "w4a4"):
            _run(
                "exp.qat.run_full_fake_quant",
                [*common, "--track", track],
            )
    else:
        _run(
            "exp.qat.run_qalora",
            [*common, "--adaptation", args.adaptation],
        )

    reg = ArtifactRegistry()
    reg.write_meta(
        RunKey(
            model=args.model,
            task=args.task,
            optimizer=args.optimizer,
            seed=args.seed,
            adaptation=args.adaptation if args.adaptation != "full_ft" else "lora_qat",
        ),
        "rq3_pipeline_complete",
    )
    _run("exp.analyze.aggregate_rq3", ["--append"])


if __name__ == "__main__":
    main()
