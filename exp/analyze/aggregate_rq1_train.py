"""Aggregate RQ1 FP-only results into results/rq1_full_ft_fp.csv."""

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
    "score_fp",
]


def _read_score(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data.get("score", data.get("metrics", {}).get("score")))


def collect_rows() -> list[dict]:
    reg = ArtifactRegistry()
    rows: list[dict] = []
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
                        rows.append(
                            {
                                "model": key.model,
                                "task": key.task,
                                "optimizer": key.optimizer,
                                "seed": key.seed,
                                "adaptation_method": "full_ft",
                                "score_fp": score_fp,
                            }
                        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    out = get_project_root() / "results" / "rq1_full_ft_fp.csv"
    rows = collect_rows()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
