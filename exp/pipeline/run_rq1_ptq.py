"""RQ1 phase B: PTQ (SmoothQuant + OmniQuant) -> quant lm-eval -> aggregate FP vs quant."""

from __future__ import annotations

import argparse
from pathlib import Path

from exp.pipeline._common import run_module
from exp.registry import ArtifactRegistry, QuantTrack, RunKey

PTQ_TRACKS: tuple[QuantTrack, ...] = ("w8a8", "w4a16", "w4a4")


def _ptq_model_ready(ptq_dir: Path) -> bool:
    if not ptq_dir.is_dir():
        return False
    if (ptq_dir / "config.json").exists():
        return True
    return any(ptq_dir.glob("*.safetensors")) or any(ptq_dir.glob("pytorch_model*.bin"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RQ1 PTQ phase: quantize FP checkpoint, evaluate each track, aggregate.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-if-done", action="store_true")
    parser.add_argument("--skip-ptq", action="store_true", help="Skip quantization steps")
    parser.add_argument("--skip-eval", action="store_true", help="Skip lm-eval on PTQ checkpoints")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument(
        "--tracks",
        default="w8a8,w4a16,w4a4",
        help="Comma-separated PTQ tracks to eval (default: all)",
    )
    parser.add_argument("--eval-backend", choices=["vllm", "hf"], default=None)
    parser.add_argument("--eval-batch-size", default=None)
    parser.add_argument("--gpu-ids", default=None, help="CUDA_VISIBLE_DEVICES for PTQ eval")
    args = parser.parse_args()

    key = RunKey(
        model=args.model,
        task=args.task,
        optimizer=args.optimizer,
        seed=args.seed,
        adaptation="full_ft",
    )
    reg = ArtifactRegistry()
    fp_dir = reg.fp_dir(key)
    if not fp_dir.exists():
        raise FileNotFoundError(
            f"FP checkpoint missing: {fp_dir}. Run RQ1 train phase first:\n"
            f"  python -m exp.pipeline.run_rq1_train --model {args.model} "
            f"--task {args.task} --optimizer {args.optimizer} --seed {args.seed}"
        )

    common = [
        "--model",
        args.model,
        "--task",
        args.task,
        "--optimizer",
        args.optimizer,
        "--seed",
        str(args.seed),
        "--adaptation",
        "full_ft",
    ]
    ptq_args = list(common)
    if args.skip_if_done:
        ptq_args.append("--skip-if-done")

    if not args.skip_ptq:
        run_module("exp.ptq.run_smoothquant", ptq_args)
        for track in ("w4a16", "w4a4"):
            run_module(
                "exp.ptq.run_omniquant",
                [*ptq_args, "--track", track],
            )

    tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]

    if not args.skip_eval:
        for track in tracks:
            if track not in PTQ_TRACKS:
                raise ValueError(f"Unknown track {track!r}; expected one of {PTQ_TRACKS}")
            ptq_dir = reg.ptq_dir(key, track)  # type: ignore[arg-type]
            metrics_path = reg.ptq_eval_metrics_path(key, track)  # type: ignore[arg-type]
            if args.skip_if_done and metrics_path.exists():
                print(f"Skip eval {track} (metrics exist)")
                continue
            if not _ptq_model_ready(ptq_dir):
                print(
                    f"Skip eval {track}: no quantized weights under {ptq_dir} "
                    f"(see RUN_*.md in that directory)"
                )
                continue
            eval_args = [
                *common,
                "--stage",
                "ptq",
                "--track",
                track,
            ]
            if args.gpu_ids:
                eval_args.extend(["--gpu-ids", args.gpu_ids])
            if args.eval_backend:
                eval_args.extend(["--backend", args.eval_backend])
            if args.eval_batch_size:
                eval_args.extend(["--batch-size", args.eval_batch_size])
            run_module("exp.eval.run_lm_eval", eval_args)

    reg.write_meta(key, "rq1_ptq_complete")

    if not args.skip_aggregate:
        run_module("exp.analyze.aggregate_rq1", [])


if __name__ == "__main__":
    main()
