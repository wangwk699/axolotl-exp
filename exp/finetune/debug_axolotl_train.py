"""Debug Axolotl training in-process (dataset load -> model -> train).

Unlike ``run_axolotl.py``, this module calls ``axolotl.cli.train.do_cli`` directly
so a single debugger session can step through the full train path without
subprocess isolation.

Recommended: use ``--num-gpus 1`` (default) so FSDP/torchrun are not required.
Multi-GPU FSDP debugging needs separate torchrun attach setup; use production
``run_axolotl`` for that.
"""

from __future__ import annotations

import argparse
import os
import sys


def _maybe_wait_for_debugger(port: int) -> None:
    import debugpy

    debugpy.listen(port)
    print(f"Waiting for debugger attach on 127.0.0.1:{port} ...", flush=True)
    debugpy.wait_for_client()
    print("Debugger attached.", flush=True)


def _run_in_process(cfg_path: str, *, optimizer: str) -> None:
    if optimizer == "shampoo":
        from exp.optimizers.shampoo_trainer import apply_shampoo_patch

        apply_shampoo_patch()
        from exp.registry import get_project_root

        root = str(get_project_root())
        os.environ["EXP_USE_SHAMPOO"] = "1"
        os.environ["PYTHONPATH"] = root + os.pathsep + os.environ.get("PYTHONPATH", "")

    from axolotl.cli.train import do_cli

    do_cli(config=cfg_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug Axolotl train in-process via axolotl.cli.train.do_cli",
    )
    parser.add_argument("--rq", type=int, required=True)
    parser.add_argument("--adaptation", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="GPUs for config rendering (default 1 for single-process debug)",
    )
    parser.add_argument("--gpu-ids", default=None)
    parser.add_argument(
        "--attach-port",
        type=int,
        default=None,
        help="If set, listen on this port and wait for VS Code/Cursor attach",
    )
    args = parser.parse_args()

    if args.num_gpus < 1:
        raise ValueError("--num-gpus must be >= 1")
    if args.num_gpus >= 2:
        print(
            "Warning: num_gpus>=2 renders FSDP multi-GPU config but this entry "
            "runs do_cli in one process. Prefer --num-gpus 1 for debugging, or "
            "use run_axolotl + torchrun attach for distributed training.",
            file=sys.stderr,
        )

    from exp.finetune.axolotl_launch import set_visible_gpus
    from exp.registry import RunKey
    from exp.render_config import render_config

    set_visible_gpus(args.gpu_ids, args.num_gpus)
    key = RunKey(
        model=args.model,
        task=args.task,
        optimizer=args.optimizer,
        seed=args.seed,
        adaptation=args.adaptation,
    )
    cfg_path = render_config(
        key,
        args.rq,
        args.adaptation,
        num_gpus=args.num_gpus,
    )
    print(f"Config: {cfg_path}", flush=True)
    print(
        "Breakpoints: axolotl/cli/train.py (do_cli, do_train, load_datasets), "
        "axolotl/train.py (train, setup_model_and_trainer)",
        flush=True,
    )

    if args.attach_port is not None:
        _maybe_wait_for_debugger(args.attach_port)

    _run_in_process(str(cfg_path), optimizer=args.optimizer)


if __name__ == "__main__":
    main()
