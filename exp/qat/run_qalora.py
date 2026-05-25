"""QA-LoRA wrapper for LoRA-QAT (external qa-lora repo)."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from exp.registry import ArtifactRegistry, RunKey, get_project_root, load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adaptation", choices=["lora", "qlora"], default="lora")
    parser.add_argument("--track", choices=["w8a8", "w4a16", "w4a4"], default="w4a16")
    parser.add_argument("--skip-if-done", action="store_true")
    args = parser.parse_args()

    key = RunKey(
        model=args.model,
        task=args.task,
        optimizer=args.optimizer,
        seed=args.seed,
        adaptation="lora_qat",
    )
    reg = ArtifactRegistry()
    stage = f"lora_qat_{args.track}"
    if args.skip_if_done and reg.stage_done(key, stage):
        return

    qa_root = get_project_root() / "external" / "qa-lora"
    if not qa_root.exists():
        raise FileNotFoundError(f"Clone qa-lora to {qa_root}")

    models = load_yaml_config("models.yaml")
    model_path = models["models"][args.model]["hf_id"]
    out_dir = reg.qat_dir(key, args.track)
    out_dir.mkdir(parents=True, exist_ok=True)

    qalora_py = qa_root / "qalora.py"
    if qalora_py.exists():
        py = os.environ.get("QALORA_PYTHON", "python")
        cmd = [py, str(qalora_py), "--model_path", model_path]
        subprocess.run(cmd, check=True, cwd=str(qa_root))

    merge_py = qa_root / "merge.py"
    readme = out_dir / "RUN_QALORA.md"
    readme.write_text(
        f"""# QA-LoRA ({args.track}) for {key.slug()}

Base model: `{model_path}`
Data: `data/processed/{args.task}/train.jsonl`

```bash
conda activate qalora
cd {qa_root}
python qalora.py --model_path {model_path}
python merge.py  # merge LoRA into GPTQ weights
cp -r <qalora_output>/* {out_dir}/
```

See https://github.com/yuhuixu1993/qa-lora
""",
        encoding="utf-8",
    )
    reg.write_meta(key, stage, extra={"readme": str(readme), "merge_script": str(merge_py)})


if __name__ == "__main__":
    main()
