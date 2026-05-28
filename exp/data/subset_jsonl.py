from __future__ import annotations

import random
from pathlib import Path


def sample_jsonl(
    src: Path,
    dst: Path,
    *,
    num_samples: int,
    seed: int,
) -> Path:
    """Sample N lines from a JSONL file without replacement.

    This is used for experiment subsetting: we keep a canonical dataset on disk,
    then materialize per-run subsets for faster iteration.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    if dst.exists():
        return dst

    lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
    if not lines:
        raise ValueError(f"Empty source dataset: {src}")

    n = min(num_samples, len(lines))
    rng = random.Random(seed)
    chosen = rng.sample(lines, n)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(chosen), encoding="utf-8")
    return dst

