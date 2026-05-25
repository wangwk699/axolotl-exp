"""Build Axolotl train commands for single- and multi-GPU launches."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def set_visible_gpus(gpu_ids: str | None, num_gpus: int) -> None:
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        return
    if "CUDA_VISIBLE_DEVICES" not in os.environ and num_gpus >= 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))


def build_train_command(
    cfg_path: Path,
    *,
    num_gpus: int,
    optimizer: str,
) -> list[str]:
    if optimizer == "shampoo":
        launcher = (
            "from exp.optimizers.shampoo_trainer import apply_shampoo_patch; "
            "apply_shampoo_patch(); "
            "import subprocess, sys; "
            f"sys.exit(subprocess.call(['axolotl', 'train', {str(cfg_path)!r}]))"
        )
        if num_gpus >= 2:
            return [
                "torchrun",
                f"--nproc_per_node={num_gpus}",
                sys.executable,
                "-c",
                launcher,
            ]
        return [sys.executable, "-c", launcher]

    if num_gpus >= 2:
        return [
            "axolotl",
            "train",
            str(cfg_path),
            "--launcher",
            "torchrun",
            "--",
            f"--nproc_per_node={num_gpus}",
        ]
    return ["axolotl", "train", str(cfg_path)]
