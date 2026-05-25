"""OmniQuant PTQ wrapper for W4A16 and W4A4."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from exp.registry import ArtifactRegistry, RunKey, get_project_root

TRACK_BITS = {
    "w4a16": (4, 16, False),
    "w4a4": (4, 4, True),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adaptation", required=True)
    parser.add_argument("--track", choices=["w4a16", "w4a4"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
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
    stage = f"ptq_{args.track}"
    if args.skip_if_done and reg.stage_done(key, stage):
        return

    wbits, abits, use_let = TRACK_BITS[args.track]
    fp_model = reg.ptq_input_dir(key)
    oq_root = get_project_root() / "external" / "OmniQuant"
    if not oq_root.exists():
        raise FileNotFoundError(
            f"Clone OmniQuant to {oq_root}. Run: bash scripts/setup_external.sh"
        )

    out_dir = reg.ptq_dir(key, args.track)
    out_dir.mkdir(parents=True, exist_ok=True)

    main_py = oq_root / "main.py"
    if main_py.exists():
        cmd = [
            os.environ.get("OMNIQUANT_PYTHON", "python"),
            str(main_py),
            "--model",
            str(fp_model),
            "--epochs",
            str(args.epochs),
            "--output_dir",
            str(out_dir),
            "--eval_ppl",
            "--wbits",
            str(wbits),
            "--abits",
            str(abits),
            "--lwc",
        ]
        if use_let:
            cmd.append("--let")
        subprocess.run(cmd, check=True, cwd=str(oq_root))
    else:
        (out_dir / "RUN_OMNIQUANT.md").write_text(
            f"""Run OmniQuant:

```bash
cd {oq_root}
CUDA_VISIBLE_DEVICES=0 python main.py \\
  --model {fp_model} \\
  --epochs {args.epochs} --output_dir {out_dir} \\
  --eval_ppl --wbits {wbits} --abits {abits} --lwc {"--let" if use_let else ""}
```
""",
            encoding="utf-8",
        )

    reg.write_meta(key, stage, parent=fp_model, extra={"track": args.track})


if __name__ == "__main__":
    main()
