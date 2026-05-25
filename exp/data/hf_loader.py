"""Load HuggingFace datasets from the local cache when available."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

DEFAULT_HF_HOME = Path("/home/wangwenkang/.cache/huggingface")

FallbackSpec = str | dict[str, str]


def get_hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", str(DEFAULT_HF_HOME)))


def ensure_hf_env() -> None:
    """Align process env with the user's shell defaults when unset."""
    os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


@contextmanager
def hf_offline(enabled: bool) -> Iterator[None]:
    previous = {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE"),
    }
    if enabled:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("HF_DATASETS_OFFLINE", None)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _repo_cache_dir(repo_id: str) -> Path:
    slug = repo_id.replace("/", "--")
    return get_hf_home() / "hub" / f"datasets--{slug}"


def _datasets_cache_dir(repo_id: str, subset: str | None) -> Path:
    slug = repo_id.replace("/", "___")
    base = get_hf_home() / "datasets" / slug
    if subset:
        return base / subset
    return base / "default"


def parse_fallbacks(
    fallbacks: list[FallbackSpec] | None,
    default_subset: str | None,
) -> list[tuple[str, str | None]]:
    parsed: list[tuple[str, str | None]] = []
    for item in fallbacks or []:
        if isinstance(item, str):
            parsed.append((item, default_subset))
        else:
            parsed.append((item["id"], item.get("subset", default_subset)))
    return parsed


def is_dataset_cached(repo_id: str, subset: str | None = None) -> bool:
    """Return True when arrow cache or hub snapshot data exists locally."""
    ds_cache = _datasets_cache_dir(repo_id, subset)
    if ds_cache.is_dir() and any(ds_cache.rglob("*.arrow")):
        return True

    hub_cache = _repo_cache_dir(repo_id)
    if not hub_cache.is_dir():
        return False

    snapshots = hub_cache / "snapshots"
    if not snapshots.is_dir():
        return False

    data_ext = {".arrow", ".parquet", ".jsonl", ".csv"}
    for path in snapshots.rglob("*"):
        if path.is_file() and path.suffix.lower() in data_ext:
            if path.stat().st_size > 0:
                return True
    return False


def any_dataset_cached(
    repo_id: str,
    subset: str | None,
    fallbacks: list[FallbackSpec] | None,
) -> bool:
    if is_dataset_cached(repo_id, subset):
        return True
    for alt_id, alt_subset in parse_fallbacks(fallbacks, subset):
        if is_dataset_cached(alt_id, alt_subset):
            return True
    return False


def load_hf_split(
    repo_id: str,
    split: str,
    *,
    subset: str | None = None,
    fallbacks: list[FallbackSpec] | None = None,
) -> Any:
    """Load one dataset split, preferring the local HF cache."""
    from datasets import DownloadMode, load_dataset

    ensure_hf_env()
    candidates: list[tuple[str, str | None]] = [(repo_id, subset)]
    candidates.extend(parse_fallbacks(fallbacks, subset))

    last_error: Exception | None = None
    for candidate_id, candidate_subset in candidates:
        candidate_kwargs: dict[str, Any] = {
            "split": split,
            "download_mode": DownloadMode.REUSE_CACHE_IF_EXISTS,
        }
        if candidate_subset:
            candidate_kwargs["name"] = candidate_subset

        if not is_dataset_cached(candidate_id, candidate_subset):
            continue

        try:
            with hf_offline(True):
                return load_dataset(candidate_id, **candidate_kwargs)
        except Exception as exc:
            last_error = exc

    online_kwargs: dict[str, Any] = {
        "split": split,
        "download_mode": DownloadMode.REUSE_CACHE_IF_EXISTS,
    }
    if subset:
        online_kwargs["name"] = subset

    try:
        with hf_offline(False):
            return load_dataset(repo_id, **online_kwargs)
    except Exception as exc:
        if last_error is not None:
            raise RuntimeError(
                f"Failed to load {repo_id} split={split} from cache or hub"
            ) from exc
        raise
