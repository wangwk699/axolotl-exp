"""Unified RunKey and artifact path registry."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

Adaptation = Literal["full_ft", "lora", "qlora", "full_qat", "lora_qat"]
QuantTrack = Literal["w8a8", "w4a16", "w4a4"]
Optimizer = Literal["adamw", "muon", "shampoo"]

PTQ_TRACK_DIRS = {
    "w8a8": "w8a8_smoothquant",
    "w4a16": "w4a16_omniquant",
    "w4a4": "w4a4_omniquant",
}


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RunKey:
    model: str
    task: str
    optimizer: str
    seed: int
    adaptation: str

    def slug(self) -> str:
        return f"{self.model}_{self.task}_{self.optimizer}_s{self.seed}_{self.adaptation}"


class ArtifactRegistry:
    def __init__(self, root: Path | None = None):
        self.root = root or (get_project_root() / "artifacts")

    def run_dir(self, key: RunKey) -> Path:
        return (
            self.root
            / key.model
            / key.task
            / key.optimizer
            / str(key.seed)
            / key.adaptation
        )

    def meta_path(self, key: RunKey) -> Path:
        return self.run_dir(key) / "meta.json"

    def fp_dir(self, key: RunKey) -> Path:
        return self.run_dir(key) / "fp"

    def lora_unmerged_dir(self, key: RunKey) -> Path:
        return self.run_dir(key) / "lora_unmerged"

    def merged_dir(self, key: RunKey) -> Path:
        return self.run_dir(key) / "merged"

    def ptq_dir(self, key: RunKey, track: QuantTrack) -> Path:
        return self.run_dir(key) / "ptq" / PTQ_TRACK_DIRS[track]

    def qat_dir(self, key: RunKey, track: QuantTrack) -> Path:
        return self.run_dir(key) / "qat" / track

    def fp_eval_metrics_path(self, key: RunKey) -> Path:
        return self.run_dir(key) / "metrics_fp.json"

    def ptq_eval_metrics_path(self, key: RunKey, track: QuantTrack) -> Path:
        return self.ptq_dir(key, track) / "metrics_quant.json"

    def qat_eval_metrics_path(self, key: RunKey, track: QuantTrack) -> Path:
        return self.qat_dir(key, track) / "metrics_quant.json"

    def ptq_input_dir(self, key: RunKey) -> Path:
        """FP checkpoint used as PTQ input (full_ft fp or merged lora/qlora)."""
        if key.adaptation == "full_ft":
            return self.fp_dir(key)
        if key.adaptation in ("lora", "qlora"):
            return self.merged_dir(key)
        raise ValueError(f"No PTQ input for adaptation={key.adaptation}")

    def ensure_dirs(self, key: RunKey) -> None:
        for p in [
            self.run_dir(key),
            self.fp_dir(key),
            self.lora_unmerged_dir(key),
            self.merged_dir(key),
        ]:
            p.mkdir(parents=True, exist_ok=True)
        for track in PTQ_TRACK_DIRS:
            self.ptq_dir(key, track).mkdir(parents=True, exist_ok=True)
            self.qat_dir(key, track).mkdir(parents=True, exist_ok=True)

    def read_meta(self, key: RunKey) -> dict[str, Any]:
        path = self.meta_path(key)
        if not path.exists():
            return {"stages": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def write_meta(
        self,
        key: RunKey,
        stage: str,
        *,
        parent: str | Path | None = None,
        axolotl_config: str | Path | None = None,
        command: str | None = None,
        metrics: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.run_dir(key).mkdir(parents=True, exist_ok=True)
        meta = self.read_meta(key)
        entry: dict[str, Any] = {
            "stage": stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_key": asdict(key),
        }
        if parent is not None:
            entry["parent"] = str(parent)
        if axolotl_config is not None:
            entry["axolotl_config"] = str(axolotl_config)
        if command is not None:
            entry["command"] = command
        if metrics is not None:
            entry["metrics"] = metrics
        if extra:
            entry.update(extra)
        stages = meta.setdefault("stages", [])
        stages.append(entry)
        self.meta_path(key).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def stage_done(self, key: RunKey, stage: str) -> bool:
        meta = self.read_meta(key)
        return any(s.get("stage") == stage for s in meta.get("stages", []))


def load_yaml_config(name: str) -> dict[str, Any]:
    import yaml

    path = get_project_root() / "configs" / name
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
