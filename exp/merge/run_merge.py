"""Merge LoRA adapters into base model via Axolotl CLI."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from exp.registry import ArtifactRegistry, RunKey
from exp.render_config import render_merge_config


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
    reg = ArtifactRegistry()
    if args.skip_if_done and reg.stage_done(key, "merged"):
        print("Skip merge (already done)")
        return

    cfg_path = render_merge_config(key)
    lora_dir = reg.lora_unmerged_dir(key)
    if not any(lora_dir.iterdir()) if lora_dir.exists() else True:
        raise FileNotFoundError(f"No LoRA checkpoint in {lora_dir}")

    cmd = [
        "axolotl",
        "merge-lora",
        str(cfg_path),
        "--lora_model_dir",
        str(lora_dir),
        "--output_dir",
        str(reg.merged_dir(key)),
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)
    reg.write_meta(
        key,
        "merged",
        parent=lora_dir,
        axolotl_config=cfg_path,
        command=" ".join(cmd),
    )


if __name__ == "__main__":
    main()
