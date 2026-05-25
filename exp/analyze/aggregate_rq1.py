"""Aggregate RQ1 results into results/rq1_full_ft_ptq.csv."""

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
]


def _read_score(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data.get("score", data.get("metrics", {}).get("score")))


def collect_rows() -> list[dict]:
    reg = ArtifactRegistry()
    rows = []
    if not reg.root.exists():
        return rows

    for model_dir in reg.root.iterdir():
        if not model_dir.is_dir():
            continue
        for task_dir in model_dir.iterdir():
            for opt_dir in task_dir.iterdir():
                for seed_dir in opt_dir.iterdir():
                    for adapt_dir in seed_dir.iterdir():
                        if adapt_dir.name != "full_ft":
                            continue
                        key = RunKey(
                            model=model_dir.name,
                            task=task_dir.name,
                            optimizer=opt_dir.name,
                            seed=int(seed_dir.name),
                            adaptation="full_ft",
                        )
                        score_fp = _read_score(reg.fp_eval_metrics_path(key))
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
                                    "adaptation_method": "full_ft",
                                    "quantization_mode": "ptq",
                                    "bitwidth_track": track,
                                    "score_fp": score_fp,
                                    "score_quant": sq,
                                    "delta_quant": score_fp - sq,
                                    "retention": sq / score_fp if score_fp else 0,
                                }
                            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    out = get_project_root() / "results" / "rq1_full_ft_ptq.csv"
    rows = collect_rows()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
