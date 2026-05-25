"""SmoothQuant W8A8 PTQ wrapper (external repo)."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from exp.registry import ArtifactRegistry, RunKey, get_project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adaptation", required=True)
    parser.add_argument("--alpha", type=float, default=0.85)
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
    out_dir = reg.ptq_dir(key, "w8a8")
    if args.skip_if_done and reg.stage_done(key, "ptq_w8a8"):
        return

    fp_model = reg.ptq_input_dir(key)
    sq_root = get_project_root() / "external" / "smoothquant"
    if not sq_root.exists():
        raise FileNotFoundError(
            f"Clone SmoothQuant to {sq_root}. Run: bash scripts/setup_external.sh"
        )

    calib = get_project_root() / "data" / "calib" / f"{args.task}.jsonl"
    scales_out = out_dir / "act_scales.pt"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: generate activation scales (uses HF model path)
    gen_script = sq_root / "examples" / "generate_act_scales.py"
    if gen_script.exists():
        cmd_gen = [
            os.environ.get("SMOOTHQUANT_PYTHON", "python"),
            str(gen_script),
            "--model-name",
            str(fp_model),
            "--output-path",
            str(scales_out),
            "--num-samples",
            "512",
            "--seq-len",
            "512",
        ]
        if calib.exists():
            cmd_gen.extend(["--dataset-path", str(calib)])
        subprocess.run(cmd_gen, check=True, cwd=str(sq_root))

    # Step 2: export quantized model (project-specific; user runs full SQ pipeline)
    readme = out_dir / "RUN_SMOOTHQUANT.md"
    readme.write_text(
        f"""# SmoothQuant W8A8 for {key.slug()}

Input FP model: `{fp_model}`
Activation scales: `{scales_out}`

Run inside the **smoothquant** conda env:

```bash
cd {sq_root}
python examples/export_int8_model.py \\
  --model-path {fp_model} \\
  --act-scales {scales_out} \\
  --alpha {args.alpha} \\
  --output {out_dir / 'int8_model'}
```

Then evaluate:

```bash
python -m exp.eval.run_lm_eval \\
  --model {key.model} --task {key.task} --optimizer {key.optimizer} \\
  --seed {key.seed} --adaptation {key.adaptation} --stage ptq --track w8a8
```

See https://github.com/mit-han-lab/smoothquant
""",
        encoding="utf-8",
    )
    reg.write_meta(
        key,
        "ptq_w8a8",
        parent=fp_model,
        extra={"output_dir": str(out_dir), "alpha": args.alpha},
    )
    print(f"Wrote {readme}")


if __name__ == "__main__":
    main()
