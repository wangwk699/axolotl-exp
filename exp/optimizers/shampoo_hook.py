"""Auto-apply Shampoo patch when imported before axolotl train."""

import os

if os.environ.get("EXP_USE_SHAMPOO") == "1":
    from exp.optimizers.shampoo_trainer import apply_shampoo_patch

    apply_shampoo_patch()
