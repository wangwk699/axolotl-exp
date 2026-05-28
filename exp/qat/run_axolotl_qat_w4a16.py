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
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--num-eval", type=int, default=1000)
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

    from exp.data.prepare_all import prepare_task
    from exp.data.subset_jsonl import sample_jsonl
    from exp.registry import get_project_root

    prepare_task(args.task, seed=args.seed)
    canonical_root = get_project_root() / "data" / "processed" / args.task
    canonical_train = canonical_root / "train.jsonl"
    canonical_eval = canonical_root / "eval.jsonl"
    ds_dir = reg.run_dir(key) / "datasets"

    train_path = sample_jsonl(
        canonical_train,
        ds_dir / f"train_n{args.num_train}_seed42.jsonl",
        num_samples=args.num_train,
        seed=42,
    )
    eval_path = sample_jsonl(
        canonical_eval,
        ds_dir / f"eval_n{args.num_eval}_seed42.jsonl",
        num_samples=args.num_eval,
        seed=42,
    )

    cfg_path = render_config(
        key,
        rq=3,
        adaptation="full_qat_w4a16",
        train_data_path=train_path,
        eval_data_path=eval_path,
    )
    cmd = ["axolotl", "train", str(cfg_path)]
    subprocess.run(cmd, check=True)
    reg.write_meta(key, "qat_w4a16", axolotl_config=cfg_path, command=" ".join(cmd))


if __name__ == "__main__":
    main()
