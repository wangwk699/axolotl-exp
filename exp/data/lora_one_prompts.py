"""
LoRA-One-compatible prompt construction.

This module mirrors the (x, y) construction in external/LoRA-One/data.py, while
keeping the implementation extensible via a small registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Tuple


Row = Dict[str, Any]
XY = Tuple[str, str]


class PromptBuilder(Protocol):
    def build(self, row: Row) -> XY: ...

    def filter_row(self, row: Row) -> bool: ...


@dataclass(frozen=True)
class SimplePromptBuilder:
    build_fn: Callable[[Row], XY]
    filter_fn: Optional[Callable[[Row], bool]] = None

    def build(self, row: Row) -> XY:
        return self.build_fn(row)

    def filter_row(self, row: Row) -> bool:
        if self.filter_fn is None:
            return True
        return bool(self.filter_fn(row))


def _template_wo_input(instruction: str) -> str:
    # Matches external/LoRA-One/data.py::template_wo_input exactly (including newlines).
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )


def _sst2_xy(row: Row) -> XY:
    instruction = "classify the sentiment of the text: "
    label_map = {0: "negative", 1: "positive", -1: "other"}
    x = f'{instruction}{row["sentence"]}\nresult: '
    y = label_map[int(row["label"])]
    return x, y


def _rte_xy(row: Row) -> XY:
    instruction = "determine if the hypothesis is entailed by the premise: "
    label_map = {0: "entailment", 1: "not_entailment", -1: "other"}
    x = (
        f"{instruction}\n"
        f'premise: {row["premise"]}\n'
        f'hypothesis: {row["hypothesis"]}\n'
        "result: "
    )
    y = label_map[int(row["label"])]
    return x, y


def _boolq_xy(row: Row) -> XY:
    instruction = "determine if the statement is true or false based on the passage: "
    label_map = {0: "false", 1: "true", -1: "other"}
    x = (
        f'{instruction}{row["passage"]}\n'
        f'Question: {row["question"]}\n'
        "result: "
    )
    y = label_map[int(row["label"])]
    return x, y


def _metamath_xy(row: Row) -> XY:
    # Mirrors external/LoRA-One/data.py::load_meta_math preprocess()
    x = f'Q: {row["query"]}\nA: '
    y = str(row["response"]).split("\nThe answer is:")[0]
    return x, y


def _codefeedback_xy(row: Row) -> XY:
    # Mirrors external/LoRA-One/data.py::load_codefeedback preprocess()
    instruction = str(row["query"])
    x = _template_wo_input(instruction=instruction)
    y_raw = str(row["answer"])
    y = "```".join(y_raw.split("```")[:2]) + "```"
    return x, y


PROMPT_BUILDERS: Dict[str, PromptBuilder] = {
    "sst2": SimplePromptBuilder(_sst2_xy),
    "rte": SimplePromptBuilder(_rte_xy),
    "boolq": SimplePromptBuilder(_boolq_xy),
    "metamath": SimplePromptBuilder(_metamath_xy),
    "codefeedback": SimplePromptBuilder(_codefeedback_xy),
}


def get_prompt_builder(task: str) -> PromptBuilder:
    try:
        return PROMPT_BUILDERS[task]
    except KeyError as exc:
        raise KeyError(
            f"Unknown task '{task}' for LoRA-One prompt building. "
            f"Add it to exp/data/lora_one_prompts.py::PROMPT_BUILDERS."
        ) from exc

