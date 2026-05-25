"""Shampoo optimizer integration for Axolotl/HF Trainer.

Set EXP_USE_SHAMPOO=1 before `axolotl train`, or import this module early:

    import exp.optimizers.shampoo_trainer  # noqa: F401
"""

from __future__ import annotations

import os
from typing import Any

from exp.registry import load_yaml_config

_PATCHED = False


def _build_shampoo(params, lr: float):
    try:
        from distributed_shampoo import DistributedShampoo
    except ImportError:
        raise ImportError(
            "Install distributed-shampoo: pip install distributed-shampoo "
            "(facebookresearch/optimizers)"
        ) from None

    cfg = load_yaml_config("optimizers.yaml")["optimizers"]["shampoo"]["shampoo"]
    return DistributedShampoo(
        params,
        lr=lr,
        max_preconditioner_dim=cfg["max_preconditioner_dim"],
        precondition_frequency=cfg["precondition_frequency"],
        start_preconditioning_step=cfg["start_preconditioning_step"],
    )


def _patch_trainer():
    global _PATCHED
    if _PATCHED:
        return
    from transformers import Trainer

    original_create_optimizer = Trainer.create_optimizer

    def create_optimizer(self: Trainer):
        if os.environ.get("EXP_USE_SHAMPOO") != "1":
            return original_create_optimizer(self)
        if self.optimizer is not None:
            return self.optimizer
        opt_model = self.model_wrapped if hasattr(self, "model_wrapped") else self.model
        params = [p for p in opt_model.parameters() if p.requires_grad]
        lr = self.args.learning_rate
        self.optimizer = _build_shampoo(params, lr)
        return self.optimizer

    Trainer.create_optimizer = create_optimizer
    _PATCHED = True


def apply_shampoo_patch() -> None:
    _patch_trainer()


if os.environ.get("EXP_USE_SHAMPOO") == "1":
    apply_shampoo_patch()
