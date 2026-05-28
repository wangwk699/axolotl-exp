"""Prepare datasets with LoRA-One-style (x, y) prompts.

This mirrors external/LoRA-One/data.py and emits Axolotl-compatible JSONL using
segment labels so the loss is only computed on the target (y).
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from exp.registry import get_project_root, load_yaml_config

Record = dict[str, Any]

LLAMA2_TOKENIZER_ID = "meta-llama/Llama-2-7b-hf"
CANONICAL_SHUFFLE_SEED = 42
CANONICAL_SAVE_TRAIN = 10_000
CANONICAL_SAVE_EVAL = 10_000


@dataclass(frozen=True)
class CanonicalManifest:
    task: str
    hf_repo: str
    hf_subset: str | None
    fallbacks: list[Any]
    shuffle_seed: int
    save_train: int
    save_eval: int
    llama2_tokenizer_id: str
    max_tokens: int
    filters: list[str]
    created_at_utc: str


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_manifest(path: Path, manifest: CanonicalManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _as_segments(x: str, y: str) -> Record:
    # LoRA-One constructs combined text as: x + " " + y + eos, with loss masked on x.
    # We encode the same thing explicitly in segments.
    return {
        "segments": [
            {"label": False, "text": x},
            {"label": True, "text": f" {y}"},
        ]
    }


def _segments_to_text(row: Record) -> Record:
    """Flatten segments for scripts that expect a single text field."""
    if "segments" in row:
        return {"text": "".join(seg["text"] for seg in row["segments"])}
    return row


def _ensure_llama2_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(LLAMA2_TOKENIZER_ID)


def _llama2_len(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def prepare_task(task: str, seed: int = 42) -> Path:
    from exp.data.hf_loader import (
        any_dataset_cached,
        ensure_hf_env,
        get_hf_home,
        load_hf_split,
    )
    from exp.data.lora_one_prompts import get_prompt_builder

    ensure_hf_env()
    cfg = load_yaml_config("tasks.yaml")["tasks"][task]
    if cfg.get("eval_only", False):
        out = get_project_root() / "data" / "processed" / task
        out.mkdir(parents=True, exist_ok=True)
        (out / "README.txt").write_text(
            f"{task} is eval-only; no SFT jsonl is generated.\n",
            encoding="utf-8",
        )
        return out

    builder = get_prompt_builder(task)
    subset = cfg.get("hf_subset")
    fallbacks = cfg.get("hf_fallback") or []
    repo_id = cfg["hf_train"]

    source = "cache" if any_dataset_cached(repo_id, subset, fallbacks) else "hub"
    print(f"  loading {task} from HF {source} ({get_hf_home()})")

    out = get_project_root() / "data" / "processed" / task
    manifest_path = out / "manifest.json"

    # Canonical LoRA-One parity path: fixed 10k train + 10k eval.
    if task in ("metamath", "codefeedback"):
        max_tokens = 512 if task == "metamath" else 1024
        filters = [
            f"llama2_len_lt_{max_tokens}",
            "shuffle_seed_42",
        ]
        if task == "metamath":
            filters.insert(0, 'type_contains_"GSM"')
        if task == "codefeedback":
            filters.insert(0, 'answer_contains_"```"')

        wanted = CanonicalManifest(
            task=task,
            hf_repo=repo_id,
            hf_subset=subset,
            fallbacks=fallbacks,
            shuffle_seed=CANONICAL_SHUFFLE_SEED,
            save_train=CANONICAL_SAVE_TRAIN,
            save_eval=CANONICAL_SAVE_EVAL,
            llama2_tokenizer_id=LLAMA2_TOKENIZER_ID,
            max_tokens=max_tokens,
            filters=filters,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        existing = _read_manifest(manifest_path)
        train_path = out / "train.jsonl"
        eval_path = out / "eval.jsonl"
        if (
            existing is not None
            and existing.get("task") == wanted.task
            and existing.get("hf_repo") == wanted.hf_repo
            and existing.get("hf_subset") == wanted.hf_subset
            and existing.get("fallbacks") == wanted.fallbacks
            and existing.get("shuffle_seed") == wanted.shuffle_seed
            and existing.get("save_train") == wanted.save_train
            and existing.get("save_eval") == wanted.save_eval
            and existing.get("llama2_tokenizer_id") == wanted.llama2_tokenizer_id
            and existing.get("max_tokens") == wanted.max_tokens
            and existing.get("filters") == wanted.filters
            and train_path.exists()
            and eval_path.exists()
        ):
            return out

        # Clear old data before writing new canonical jsonl.
        if out.exists():
            shutil.rmtree(out)
        calib_path = get_project_root() / "data" / "calib" / f"{task}.jsonl"
        if calib_path.exists():
            calib_path.unlink()

        train_ds = load_hf_split(
            repo_id,
            cfg["split_train"],
            subset=subset,
            fallbacks=fallbacks,
        )
        tok = _ensure_llama2_tokenizer()
        all_rows: list[Record] = []
        for r in train_ds:
            if not builder.filter_row(r):
                continue
            if task == "metamath":
                if "GSM" not in str(r.get("type", "")):
                    continue
            else:
                if "```" not in str(r.get("answer", "")):
                    continue

            x, y = builder.build(r)
            if _llama2_len(tok, f"{x} {y}") >= max_tokens:
                continue
            all_rows.append(_as_segments(x, y))

        rng = random.Random(CANONICAL_SHUFFLE_SEED)
        rng.shuffle(all_rows)
        train_rows = all_rows[:CANONICAL_SAVE_TRAIN]
        eval_rows = all_rows[
            CANONICAL_SAVE_TRAIN : CANONICAL_SAVE_TRAIN + CANONICAL_SAVE_EVAL
        ]

        _write_jsonl(out / "train.jsonl", train_rows)
        _write_jsonl(out / "eval.jsonl", eval_rows)
        _write_manifest(manifest_path, wanted)

        # calib stays compatible with external PTQ/QAT scripts.
        calib_src = train_rows if train_rows else eval_rows
        calib_n = load_yaml_config("tasks.yaml")["calib"]["num_samples"]
        calib = rng.sample(calib_src, min(calib_n, len(calib_src)))
        calib = [_segments_to_text(r) for r in calib]
        calib_root = get_project_root() / "data" / "calib"
        _write_jsonl(calib_root / f"{task}.jsonl", calib)
        return out

    train_ds = load_hf_split(
        repo_id,
        cfg["split_train"],
        subset=subset,
        fallbacks=fallbacks,
    )
    if cfg.get("eval_from_train", False):
        from datasets import Dataset

        if not isinstance(train_ds, Dataset):
            raise TypeError(
                f"{task}: eval_from_train requires an in-memory datasets.Dataset"
            )
        eval_size = int(cfg.get("eval_size", 10_000))
        if eval_size <= 0:
            raise ValueError(f"{task}: eval_size must be > 0")
        split = train_ds.train_test_split(
            test_size=min(eval_size, len(train_ds)),
            shuffle=True,
            seed=seed,
        )
        train_ds = split["train"]
        eval_ds = split["test"]
    else:
        eval_ds = load_hf_split(
            cfg["hf_eval"],
            cfg["split_eval"],
            subset=subset,
            fallbacks=fallbacks,
        )

    def convert_row(r: dict[str, Any]) -> Record:
        if not builder.filter_row(r):
            return {}
        x, y = builder.build(r)
        return _as_segments(x, y)

    train_rows = [row for r in train_ds if (row := convert_row(r))]
    eval_rows = [row for r in eval_ds if (row := convert_row(r))]

    _write_jsonl(out / "train.jsonl", train_rows)
    _write_jsonl(out / "eval.jsonl", eval_rows)

    rng = random.Random(seed)
    calib_src = train_rows if train_rows else eval_rows
    calib_n = load_yaml_config("tasks.yaml")["calib"]["num_samples"]
    calib = rng.sample(calib_src, min(calib_n, len(calib_src)))
    # Export calib as flat text for external PTQ/QAT scripts.
    calib = [_segments_to_text(r) for r in calib]
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
