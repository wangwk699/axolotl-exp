"""Convert downstream tasks to Axolotl-compatible alpaca JSONL."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable

from exp.registry import get_project_root, load_yaml_config

Record = dict[str, Any]


def _write_jsonl(path: Path, rows: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sst2(row: dict[str, Any]) -> Record:
    label = "positive" if row["label"] == 1 else "negative"
    return {
        "instruction": "Classify the sentiment of the sentence as positive or negative.",
        "input": row["text"],
        "output": label,
    }


def _rte(row: dict[str, Any]) -> Record:
    label = "entailment" if row["label"] == 0 else "not_entailment"
    return {
        "instruction": "Determine whether the hypothesis is entailed by the premise.",
        "input": f"Premise: {row['sentence1']}\nHypothesis: {row['sentence2']}",
        "output": label,
    }


def _boolq(row: dict[str, Any]) -> Record:
    label = "yes" if row["answer"] else "no"
    return {
        "instruction": "Answer the question with yes or no based on the passage.",
        "input": f"Passage: {row['passage']}\nQuestion: {row['question']}",
        "output": label,
    }


def _multirc(row: dict[str, Any]) -> Record:
    label = "yes" if row["label"] == 1 else "no"
    return {
        "instruction": "Answer whether the answer option correctly answers the question given the paragraph.",
        "input": (
            f"Paragraph: {row['paragraph']}\n"
            f"Question: {row['question']}\n"
            f"Answer: {row['answer']}"
        ),
        "output": label,
    }


def _gsm8k(row: dict[str, Any]) -> Record:
    """Match lm-eval gsm8k: Question/Answer completion with full CoT targets."""
    prompt = f"Question: {row['question']}\nAnswer:"
    return {
        "segments": [
            {"label": False, "text": prompt},
            {"label": True, "text": f" {row['answer']}"},
        ],
    }


def _gsm8k_calib_text(row: Record) -> Record:
    """Flat text for PTQ calib scripts that expect a single text field."""
    if "segments" in row:
        return {"text": "".join(seg["text"] for seg in row["segments"])}
    return row


def _commonsenseqa(row: dict[str, Any]) -> Record:
    choices = row["choices"]
    labels = choices["label"]
    texts = choices["text"]
    idx = labels.index(row["answerKey"])
    return {
        "instruction": "Choose the best answer to the commonsense question.",
        "input": f"Question: {row['question']}\nChoices: " + " | ".join(
            f"{l}. {t}" for l, t in zip(labels, texts)
        ),
        "output": texts[idx],
    }


CONVERTERS: dict[str, Callable[[dict[str, Any]], Record]] = {
    "sst2": _sst2,
    "rte": _rte,
    "boolq": _boolq,
    "multirc": _multirc,
    "gsm8k": _gsm8k,
    "commonsenseqa": _commonsenseqa,
}


def prepare_task(task: str, seed: int = 42) -> Path:
    from exp.data.hf_loader import (
        any_dataset_cached,
        ensure_hf_env,
        get_hf_home,
        load_hf_split,
    )

    ensure_hf_env()
    cfg = load_yaml_config("tasks.yaml")["tasks"][task]
    if task == "humaneval":
        out = get_project_root() / "data" / "processed" / task
        out.mkdir(parents=True, exist_ok=True)
        (out / "README.txt").write_text(
            "HumanEval uses lm-eval only; no SFT jsonl required.\n",
            encoding="utf-8",
        )
        return out

    converter = CONVERTERS[task]
    subset = cfg.get("hf_subset")
    fallbacks = cfg.get("hf_fallback") or []
    repo_id = cfg["hf_train"]

    source = "cache" if any_dataset_cached(repo_id, subset, fallbacks) else "hub"
    print(f"  loading {task} from HF {source} ({get_hf_home()})")

    train_ds = load_hf_split(
        repo_id,
        cfg["split_train"],
        subset=subset,
        fallbacks=fallbacks,
    )
    eval_ds = load_hf_split(
        cfg["hf_eval"],
        cfg["split_eval"],
        subset=subset,
        fallbacks=fallbacks,
    )

    train_rows = [converter(r) for r in train_ds]
    eval_rows = [converter(r) for r in eval_ds]

    out = get_project_root() / "data" / "processed" / task
    _write_jsonl(out / "train.jsonl", train_rows)
    _write_jsonl(out / "eval.jsonl", eval_rows)

    rng = random.Random(seed)
    calib_src = train_rows if train_rows else eval_rows
    calib_n = load_yaml_config("tasks.yaml")["calib"]["num_samples"]
    calib = rng.sample(calib_src, min(calib_n, len(calib_src)))
    if task == "gsm8k":
        calib = [_gsm8k_calib_text(r) for r in calib]
    calib_root = get_project_root() / "data" / "calib"
    _write_jsonl(calib_root / f"{task}.jsonl", calib)
    return out


def prepare_all(seed: int = 42) -> None:
    for task in load_yaml_config("tasks.yaml")["task_list"]:
        try:
            prepare_task(task, seed=seed)
            print(f"Prepared {task}")
        except Exception as exc:
            print(f"FAILED {task}: {exc}")


if __name__ == "__main__":
    prepare_all()
