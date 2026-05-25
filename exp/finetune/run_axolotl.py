"""Run Axolotl fine-tuning with optional Shampoo hook."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

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
    parser.add_argument("--skip-if-done", action="store_true")
    args = parser.parse_args()

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

    cfg_path = render_config(key, args.rq, args.adaptation)
    env = os.environ.copy()
    if args.optimizer == "shampoo":
        env["EXP_USE_SHAMPOO"] = "1"
        root = str(get_project_root())
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        launcher = (
            "from exp.optimizers.shampoo_trainer import apply_shampoo_patch; "
            "apply_shampoo_patch(); "
            "import subprocess, sys; "
            f"sys.exit(subprocess.call(['axolotl', 'train', {str(cfg_path)!r}]))"
        )
        cmd = [sys.executable, "-c", launcher]
    else:
        cmd = ["axolotl", "train", str(cfg_path)]

    _run(cmd, env=env)
    reg.write_meta(
        key,
        stage,
        axolotl_config=cfg_path,
        command=" ".join(cmd),
    )


if __name__ == "__main__":
    main()
