from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .io import stable_fraction


def stable_split(records: Iterable[dict[str, Any]], id_field: str, seed: int, validation_fraction: float):
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for record in records:
        identifier = str(record[id_field])
        target = validation if stable_fraction(identifier, seed) < validation_fraction else train
        target.append(record)
    return train, validation


def stratified_holdout(
    records: Iterable[dict[str, Any]],
    id_field: str,
    stratum_field: str,
    seed: int,
    development_fraction: float,
):
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(stratum_field, "UNKNOWN"))].append(record)
    development: list[dict[str, Any]] = []
    final_test: list[dict[str, Any]] = []
    for group in groups.values():
        ordered = sorted(group, key=lambda item: stable_fraction(str(item[id_field]), seed))
        count = max(1, round(len(ordered) * development_fraction)) if len(ordered) > 1 else 0
        development.extend(ordered[:count])
        final_test.extend(ordered[count:])
    return development, final_test


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

