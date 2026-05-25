"""Aggregate RQ2 results into results/rq2_lora_ft_ptq.csv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from exp.registry import ArtifactRegistry, RunKey, get_project_root

COLUMNS = [
    "model",
    "task",
    "optimizer",
    "seed",
    "adaptation_method",
    "quantization_mode",
    "bitwidth_track",
    "score_fp",
    "score_quant",
    "delta_quant",
    "retention",
    "score_lora_unmerged",
    "score_lora_merged",
]


def _read_score(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data.get("score"))


def collect_rows() -> list[dict]:
    reg = ArtifactRegistry()
    rows = []
    for model_dir in reg.root.iterdir() if reg.root.exists() else []:
        for task_dir in model_dir.iterdir():
            for opt_dir in task_dir.iterdir():
                for seed_dir in opt_dir.iterdir():
                    for adapt_dir in seed_dir.iterdir():
                        if adapt_dir.name not in ("lora", "qlora"):
                            continue
                        key = RunKey(
                            model=model_dir.name,
                            task=task_dir.name,
                            optimizer=opt_dir.name,
                            seed=int(seed_dir.name),
                            adaptation=adapt_dir.name,
                        )
                        score_unmerged = _read_score(
                            reg.run_dir(key) / "metrics_lora_unmerged.json"
                        )
                        score_merged = _read_score(
                            reg.run_dir(key) / "metrics_lora_merged.json"
                        )
                        score_fp = score_merged
                        if score_fp is None:
                            continue
                        for track in ("w8a8", "w4a16", "w4a4"):
                            sq = _read_score(reg.ptq_eval_metrics_path(key, track))
                            if sq is None:
                                continue
                            rows.append(
                                {
                                    "model": key.model,
                                    "task": key.task,
                                    "optimizer": key.optimizer,
                                    "seed": key.seed,
                                    "adaptation_method": key.adaptation,
                                    "quantization_mode": "ptq",
                                    "bitwidth_track": track,
                                    "score_fp": score_fp,
                                    "score_quant": sq,
                                    "delta_quant": score_fp - sq,
                                    "retention": sq / score_fp if score_fp else 0,
                                    "score_lora_unmerged": score_unmerged,
                                    "score_lora_merged": score_merged,
                                }
                            )
    return rows


def main() -> None:
    out = get_project_root() / "results" / "rq2_lora_ft_ptq.csv"
    rows = collect_rows()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
