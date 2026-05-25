"""Axolotl W4A16 weight-only QAT (Full-FT)."""

from __future__ import annotations

import argparse
import subprocess

from exp.registry import ArtifactRegistry, RunKey
from exp.render_config import render_config


def main() -> None:
    parser = argparse.ArgumentParser()
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
        adaptation="full_ft",
    )
    reg = ArtifactRegistry()
    if args.skip_if_done and reg.stage_done(key, "qat_w4a16"):
        return

    cfg_path = render_config(key, rq=3, adaptation="full_qat_w4a16")
    cmd = ["axolotl", "train", str(cfg_path)]
    subprocess.run(cmd, check=True)
    reg.write_meta(key, "qat_w4a16", axolotl_config=cfg_path, command=" ".join(cmd))


if __name__ == "__main__":
    main()
