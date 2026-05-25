"""Run Axolotl fine-tuning with optional Shampoo hook."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from exp.finetune.axolotl_launch import build_train_command, set_visible_gpus
from exp.registry import ArtifactRegistry, RunKey, get_project_root
from exp.render_config import render_config


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env or os.environ.copy())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rq", type=int, required=True)
    parser.add_argument("--adaptation", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated GPU ids (e.g. 0,1). Defaults to 0..N-1 when unset.",
    )
    parser.add_argument("--skip-if-done", action="store_true")
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
    cfg_path = render_config(
        key,
        args.rq,
        args.adaptation,
        num_gpus=args.num_gpus,
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


if __name__ == "__main__":
    main()
