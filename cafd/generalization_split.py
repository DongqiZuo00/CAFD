"""Frozen nested-validation split for the Mistral CAFD generalization audit."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .data import _write_jsonl, load_rows
from .training_common import write_json


SPLIT_SEED = 12027
SELECTION_COUNTS = {
    "contains_count": 14,
    "contains_ordered": 25,
    "contains_substring": 25,
}


def _ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["id"]) for row in rows]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare(root: Path) -> dict[str, Any]:
    root = root.resolve()
    source = load_rows(root, "train")
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source:
        by_family[str(row["problem_family"])].append(row)
    if set(by_family) != set(SELECTION_COUNTS):
        raise RuntimeError(f"unexpected HAS families: {sorted(by_family)}")

    rng = random.Random(SPLIT_SEED)
    selection: list[dict[str, Any]] = []
    for family, count in SELECTION_COUNTS.items():
        candidates = list(by_family[family])
        rng.shuffle(candidates)
        if len(candidates) <= count:
            raise RuntimeError(f"insufficient {family} rows for nested validation")
        selection.extend(candidates[:count])
    rng.shuffle(selection)
    selection_ids = set(_ids(selection))
    fit = [row for row in source if str(row["id"]) not in selection_ids]
    if len(selection) != 64 or len(fit) != 614:
        raise RuntimeError(f"bad nested split sizes: fit={len(fit)} selection={len(selection)}")
    if selection_ids & set(_ids(fit)):
        raise RuntimeError("nested fit and selection IDs overlap")
    if set(_ids(source)) != selection_ids | set(_ids(fit)):
        raise RuntimeError("nested split is not an exact partition of CAFD train")

    output = root / "data" / "cafd" / "generalization_v10"
    fit_path = output / "fit.jsonl"
    selection_path = output / "selection.jsonl"
    manifest_path = output / "manifest.json"
    expected = {
        "status": "frozen",
        "seed": SPLIT_SEED,
        "source": "data/cafd/train.jsonl",
        "fit_count": len(fit),
        "selection_count": len(selection),
        "selection_family_counts": dict(sorted(Counter(
            str(row["problem_family"]) for row in selection
        ).items())),
        "fit_ids": _ids(fit),
        "selection_ids": _ids(selection),
        "confirmation_split": "data/cafd/development.jsonl",
        "frozen_test_accessed": False,
    }
    if manifest_path.exists():
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise RuntimeError("nested-validation split is already frozen differently")
        if _ids(_read_jsonl(fit_path)) != expected["fit_ids"]:
            raise RuntimeError("frozen fit split changed")
        if _ids(_read_jsonl(selection_path)) != expected["selection_ids"]:
            raise RuntimeError("frozen selection split changed")
        return expected
    _write_jsonl(fit_path, fit)
    _write_jsonl(selection_path, selection)
    write_json(manifest_path, expected)
    return expected


def load_nested_rows(root: Path, split: str) -> list[dict[str, Any]]:
    if split not in {"fit", "selection"}:
        raise ValueError(split)
    path = root.resolve() / "data" / "cafd" / "generalization_v10" / f"{split}.jsonl"
    return _read_jsonl(path)
