"""Deterministic verified canonical programs for the three Manufactoria-HAS families."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Callable, Hashable

from transformers import PreTrainedTokenizerBase

from .configuration import load_experiment_config
from .data import bounded_prompt, load_rows
from .layout import experiment_layout
from .modeling import load_tokenizers
from .prompting import prompt_token_ids
from .training_common import write_json
from .verifier import score_completion


COLORS = ("R", "B", "Y", "G")


def _target(state: Hashable, accepting: Callable[[Hashable], bool], names: dict[Hashable, int]) -> str:
    return "end" if accepting(state) else f"rb_{names[state]}"


def automaton_program(
    states: list[Hashable],
    start_state: Hashable,
    transition: Callable[[Hashable, str], Hashable],
    accepting: Callable[[Hashable], bool],
) -> str:
    """Compile a four-symbol DFA to the official two-puller Manufactoria DSL."""

    active = [state for state in states if not accepting(state)]
    names = {state: index for index, state in enumerate(active)}
    if accepting(start_state):
        start_target = "end"
    else:
        start_target = f"rb_{names[start_state]}"
    lines = ["START start:", f"    NEXT {start_target}", ""]
    for state in active:
        index = names[state]
        lines.extend(
            [
                f"PULLER_RB rb_{index}:",
                f"    [R] {_target(transition(state, 'R'), accepting, names)}",
                f"    [B] {_target(transition(state, 'B'), accepting, names)}",
                f"    [EMPTY] yg_{index}",
                "",
                f"PULLER_YG yg_{index}:",
                f"    [Y] {_target(transition(state, 'Y'), accepting, names)}",
                f"    [G] {_target(transition(state, 'G'), accepting, names)}",
                "    [EMPTY] NONE",
                "",
            ]
        )
    lines.append("END end")
    return "\n".join(lines)


def _ordered(pattern: str) -> str:
    states = list(range(len(pattern) + 1))

    def transition(state: int, color: str) -> int:
        return state + 1 if state < len(pattern) and color == pattern[state] else state

    return automaton_program(states, 0, transition, lambda state: state == len(pattern))


def _substring(pattern: str) -> str:
    states = list(range(len(pattern) + 1))

    def transition(state: int, color: str) -> int:
        candidate = pattern[:state] + color
        for length in range(min(len(pattern), len(candidate)), -1, -1):
            if candidate.endswith(pattern[:length]):
                return length
        raise AssertionError("empty prefix must match")

    return automaton_program(states, 0, transition, lambda state: state == len(pattern))


def _count(requirements: list[tuple[str, int]]) -> str:
    colors = [color for color, _ in requirements]
    limits = [limit for _, limit in requirements]
    states = list(itertools.product(*(range(limit + 1) for limit in limits)))

    def transition(state: tuple[int, ...], color: str) -> tuple[int, ...]:
        values = list(state)
        if color in colors:
            index = colors.index(color)
            values[index] = min(values[index] + 1, limits[index])
        return tuple(values)

    return automaton_program(states, tuple(0 for _ in limits), transition, lambda state: all(a >= b for a, b in zip(state, limits, strict=True)))


def canonical_program(row: dict) -> str:
    family = row["problem_family"]
    name = row["name"]
    if family == "contains_ordered":
        pattern = name.removeprefix("Contains ").strip()
        return _ordered(pattern)
    if family == "contains_substring":
        pattern = name.removeprefix("Contains ").strip().strip("'")
        return _substring(pattern)
    if family == "contains_count":
        requirements = [(color, int(count)) for count, color in re.findall(r"(\d+)\s+([RBYG])(?:s)?", name)]
        if not requirements:
            raise ValueError(f"cannot parse count requirements from {name!r}")
        return _count(requirements)
    raise ValueError(f"unsupported HAS family: {family}")


def fenced(program: str) -> str:
    return f"```manufactoria\n{program.rstrip()}\n```"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare(root: Path, tokenizer: PreTrainedTokenizerBase) -> dict[str, int]:
    config = load_experiment_config(root)
    layout = experiment_layout(root, config)
    renderer = str(config["generation"]["prompt_renderer"])
    training_rows = load_rows(root, "train")
    development_rows = load_rows(root, "development")
    verified_training: list[dict] = []
    maximum_tokens = 0
    maximum_task = ""
    for split, rows in (("train", training_rows), ("development", development_rows)):
        for row in rows:
            completion = fenced(canonical_program(row))
            if score_completion(completion, row["ground_truth"], "full_pass", require_contract=True) != 1.0:
                raise RuntimeError(f"canonical program failed official verifier: {split}/{row['id']}")
            completion_ids = tokenizer.encode(completion, add_special_tokens=False)
            length = len(completion_ids)
            if length > maximum_tokens:
                maximum_tokens = length
                maximum_task = row["id"]
            if split == "train":
                prompt_ids = prompt_token_ids(tokenizer, bounded_prompt(row), renderer=renderer)
                verified_training.append(
                    {
                        "id": row["id"],
                        "prompt_ids": prompt_ids,
                        "completion_ids": completion_ids + [tokenizer.eos_token_id],
                        "source": "deterministic-canonical-verified",
                    }
                )
    max_new_tokens = 1024 if maximum_tokens <= 768 else 2048
    output = layout.capacity / "verified_train_solutions.jsonl"
    _write_jsonl(output, verified_training)
    contract = {
        "status": "fixed",
        "max_canonical_tokens": maximum_tokens,
        "max_canonical_task": maximum_task,
        "max_new_tokens": max_new_tokens,
        "response_contract": "exactly one Manufactoria DSL code block; stop after closing fence",
        "verified_training_solutions": len(verified_training),
    }
    write_json(layout.state_root / "generation_contract.json", contract)
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    _, student_tokenizer = load_tokenizers(root / ".cache" / "huggingface" / "hub")
    print(json.dumps(prepare(root, student_tokenizer), sort_keys=True))


if __name__ == "__main__":
    main()
