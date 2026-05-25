"""Aggregate RQ3 QAT results and PTQ vs QAT rank correlation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from exp.registry import ArtifactRegistry, RunKey, get_project_root

QAT_COLUMNS = [
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

CORR_COLUMNS = [
    "model",
    "task",
    "adaptation_method",
    "bitwidth_track",
    "rank_ptq",
    "rank_qat",
    "kendall_tau",
    "spearman_rho",
]


def _read_score(path: Path) -> float | None:
    if not path.exists():
        return None
    return float(json.loads(path.read_text(encoding="utf-8")).get("score"))


def collect_qat_rows() -> list[dict]:
    reg = ArtifactRegistry()
    rows = []
    for model_dir in reg.root.iterdir() if reg.root.exists() else []:
        for task_dir in model_dir.iterdir():
            for opt_dir in task_dir.iterdir():
                for seed_dir in opt_dir.iterdir():
                    for adapt_dir in seed_dir.iterdir():
                        key = RunKey(
                            model=model_dir.name,
                            task=task_dir.name,
                            optimizer=opt_dir.name,
                            seed=int(seed_dir.name),
                            adaptation=adapt_dir.name,
                        )
                        score_fp = _read_score(reg.fp_eval_metrics_path(key))
                        if score_fp is None:
                            score_fp = _read_score(
                                reg.run_dir(key) / "metrics_lora_merged.json"
                            )
                        for track in ("w8a8", "w4a16", "w4a4"):
                            sq = _read_score(reg.qat_eval_metrics_path(key, track))
                            if sq is None or score_fp is None:
                                continue
                            rows.append(
                                {
                                    "model": key.model,
                                    "task": key.task,
                                    "optimizer": key.optimizer,
                                    "seed": key.seed,
                                    "adaptation_method": key.adaptation,
                                    "quantization_mode": "qat",
                                    "bitwidth_track": track,
                                    "score_fp": score_fp,
                                    "score_quant": sq,
                                    "delta_quant": score_fp - sq,
                                    "retention": sq / score_fp if score_fp else 0,
                                }
                            )
    return rows


def _rank_dict(scores: dict[str, float]) -> dict[str, int]:
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return {k: i + 1 for i, (k, _) in enumerate(ordered)}


def collect_correlations() -> list[dict]:
    root = get_project_root() / "results"
    rq1_path = root / "rq1_full_ft_ptq.csv"
    rq2_path = root / "rq2_lora_ft_ptq.csv"
    qat_rows = collect_qat_rows()

    ptq_rows = []
    for p in (rq1_path, rq2_path):
        if p.exists():
            with p.open(encoding="utf-8") as f:
                ptq_rows.extend(list(csv.DictReader(f)))

    corr_rows = []
    optimizers = ["adamw", "muon", "shampoo"]
    group_keys = set(
        (r["model"], r["task"], r["adaptation_method"], r["bitwidth_track"])
        for r in ptq_rows + qat_rows
    )

    for model, task, adaptation, track in group_keys:
        ptq_scores = {
            r["optimizer"]: float(r["score_quant"])
            for r in ptq_rows
            if r["model"] == model
            and r["task"] == task
            and r["adaptation_method"] == adaptation
            and r["bitwidth_track"] == track
            and r["optimizer"] in optimizers
        }
        qat_scores = {
            r["optimizer"]: float(r["score_quant"])
            for r in qat_rows
            if r["model"] == model
            and r["task"] == task
            and r["adaptation_method"] == adaptation
            and r["bitwidth_track"] == track
            and r["optimizer"] in optimizers
        }
        common = [o for o in optimizers if o in ptq_scores and o in qat_scores]
        if len(common) < 2:
            continue
        ptq_r = _rank_dict({o: ptq_scores[o] for o in common})
        qat_r = _rank_dict({o: qat_scores[o] for o in common})
        ptq_vals = [ptq_r[o] for o in common]
        qat_vals = [qat_r[o] for o in common]
        kt = kendalltau(ptq_vals, qat_vals).statistic
        sp = spearmanr(ptq_vals, qat_vals).statistic
        corr_rows.append(
            {
                "model": model,
                "task": task,
                "adaptation_method": adaptation,
                "bitwidth_track": track,
                "rank_ptq": str(ptq_r),
                "rank_qat": str(qat_r),
                "kendall_tau": kt,
                "spearman_rho": sp,
            }
        )
    return corr_rows


def main() -> None:
    root = get_project_root() / "results"
    root.mkdir(parents=True, exist_ok=True)

    qat_rows = collect_qat_rows()
    qat_out = root / "rq3_qat.csv"
    with qat_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=QAT_COLUMNS)
        w.writeheader()
        w.writerows(qat_rows)

    corr_rows = collect_correlations()
    corr_out = root / "rq3_ptq_qat_rank_correlation.csv"
    with corr_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CORR_COLUMNS)
        w.writeheader()
        w.writerows(corr_rows)

    print(f"Wrote {len(qat_rows)} QAT rows, {len(corr_rows)} correlation rows")


if __name__ == "__main__":
    main()
