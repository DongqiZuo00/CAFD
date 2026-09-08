"""Pinned Manufactoria-HAS data materialization and frozen split handling."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


TRAIN_REPO = "manufactoria/has_train"
TRAIN_REVISION = "c4be8c1715810b7e5afd68869e23cf04e9dca7d6"
TEST_REPO = "manufactoria/has_test"
TEST_REVISION = "13d4b794afe41f34ffed126c34faacf00c445349"
SEED = 2027
DEV_SIZE = 64


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _rows(dataset: Dataset) -> list[dict[str, Any]]:
    return [dict(dataset[index]) for index in range(len(dataset))]


def prepare(root: Path) -> dict[str, int]:
    output = root / "data" / "cafd"
    train_source = load_dataset(TRAIN_REPO, split="train", revision=TRAIN_REVISION)
    test_source = load_dataset(TEST_REPO, split="train", revision=TEST_REVISION)
    if len(train_source) != 742 or len(test_source) != 132:
        raise RuntimeError(f"unexpected official split sizes: train={len(train_source)}, test={len(test_source)}")

    source_rows = _rows(train_source)
    test_rows = _rows(test_source)
    indices = list(range(len(source_rows)))
    random.Random(SEED).shuffle(indices)
    dev_indices = set(indices[:DEV_SIZE])
    dev_rows = [source_rows[index] for index in indices[:DEV_SIZE]]
    train_rows = [row for index, row in enumerate(source_rows) if index not in dev_indices]

    train_ids = {row["id"] for row in train_rows}
    dev_ids = {row["id"] for row in dev_rows}
    test_ids = {row["id"] for row in test_rows}
    if train_ids & dev_ids or train_ids & test_ids or dev_ids & test_ids:
        raise RuntimeError("official test or development tasks entered the training split")
    if len(train_ids) != len(train_rows) or len(dev_ids) != len(dev_rows) or len(test_ids) != len(test_rows):
        raise RuntimeError("duplicate task IDs in frozen splits")

    _write_jsonl(output / "train.jsonl", train_rows)
    _write_jsonl(output / "development.jsonl", dev_rows)
    _write_jsonl(output / "test.jsonl", test_rows)
    return {"train": len(train_rows), "development": len(dev_rows), "test": len(test_rows)}


def load_rows(root: Path, split: str) -> list[dict[str, Any]]:
    path = root / "data" / "cafd" / f"{split}.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError(f"task {row.get('id')} does not use the DELTA canonical single-user prompt")
    return str(messages[0]["content"])


RESPONSE_CONTRACT = """

# Response Contract
Return exactly one Manufactoria DSL code block and nothing else.
The first output characters must be ```manufactoria followed by a newline.
Do not output analysis, reasoning, explanations, or a <think> block.
Close the single code block with ``` and stop immediately after the closing fence.
""".rstrip()


def bounded_prompt(row: dict[str, Any]) -> str:
    """The single plain-text renderer shared by Teacher and Student."""

    return canonical_prompt(row).rstrip() + RESPONSE_CONTRACT


def prompt_stream(rows: list[dict[str, Any]], updates: int, prompts_per_update: int, seed: int = SEED) -> list[list[int]]:
    """One deterministic stream shared byte-for-byte by all Student methods."""

    rng = random.Random(seed)
    order: list[int] = []
    while len(order) < updates * prompts_per_update:
        epoch = list(range(len(rows)))
        rng.shuffle(epoch)
        order.extend(epoch)
    return [order[i * prompts_per_update : (i + 1) * prompts_per_update] for i in range(updates)]


def rollout_seed(update: int, prompt_slot: int, sample_slot: int, base_seed: int = SEED) -> int:
    return base_seed * 1_000_003 + update * 10_007 + prompt_slot * 101 + sample_slot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    counts = prepare(args.root.resolve())
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
