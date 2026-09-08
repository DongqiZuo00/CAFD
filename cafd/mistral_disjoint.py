"""Deterministic disjoint Teacher SFT and RLVR pools for Mistral CAFD-v7."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Sequence

from .configuration import load_experiment_config
from .data import load_rows
from .layout import experiment_layout
from .mistral_runtime import V7_PROTOCOL, validate_config
from .training_common import checkpoint_complete, write_json


EXPECTED_TRAINING_TASKS = 678


def deterministic_partition(
    task_ids: Sequence[str],
    *,
    seed: int,
    sft_count: int,
) -> tuple[list[str], list[str]]:
    ordered = [str(task_id) for task_id in task_ids]
    if len(ordered) != EXPECTED_TRAINING_TASKS:
        raise RuntimeError(
            f"training task count changed: {len(ordered)} != {EXPECTED_TRAINING_TASKS}"
        )
    if len(set(ordered)) != len(ordered):
        raise RuntimeError("duplicate task IDs in Teacher partition source")
    if not 0 < sft_count < len(ordered):
        raise RuntimeError(f"invalid Teacher SFT count: {sft_count}")
    shuffled = ordered.copy()
    random.Random(seed).shuffle(shuffled)
    sft_set = set(shuffled[:sft_count])
    sft_ids = [task_id for task_id in ordered if task_id in sft_set]
    rlvr_ids = [task_id for task_id in ordered if task_id not in sft_set]
    if set(sft_ids) & set(rlvr_ids) or set(sft_ids) | set(rlvr_ids) != set(ordered):
        raise RuntimeError("Teacher SFT/RLVR partition is not an exact disjoint union")
    return sft_ids, rlvr_ids


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def teacher_rlvr_rows(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    if str(config["protocol"]) != V7_PROTOCOL:
        return load_rows(root, "train")
    layout = experiment_layout(root, config)
    manifest_path = layout.run_root / "teacher_data_split.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen":
        raise RuntimeError(f"Teacher data split is not frozen: {manifest}")
    expected = [str(task_id) for task_id in manifest["teacher_rlvr_ids"]]
    rows = load_rows(root, "train")
    by_id = {str(row["id"]): row for row in rows}
    if len(by_id) != len(rows) or set(expected) - set(by_id):
        raise RuntimeError("Teacher RLVR manifest does not match the training split")
    selected = [by_id[task_id] for task_id in expected]
    if len(selected) != int(manifest["teacher_rlvr_count"]):
        raise RuntimeError("Teacher RLVR materialization count changed")
    return selected


def prepare_disjoint(root: Path) -> None:
    config = load_experiment_config(root)
    validate_config(config)
    if str(config["protocol"]) != V7_PROTOCOL:
        raise RuntimeError("disjoint preparation is only valid for Mistral CAFD-v7")
    layout = experiment_layout(root, config)
    settings = config["data_split"]
    sft_count = int(settings["teacher_sft_solutions"])
    rlvr_count = int(settings["teacher_rlvr_prompts"])
    if sft_count + rlvr_count != EXPECTED_TRAINING_TASKS:
        raise RuntimeError("Teacher partition counts do not cover all training tasks")

    training_rows = load_rows(root, "train")
    ordered_ids = [str(row["id"]) for row in training_rows]
    sft_ids, rlvr_ids = deterministic_partition(
        ordered_ids,
        seed=int(settings["split_seed"]),
        sft_count=sft_count,
    )
    if len(sft_ids) != sft_count or len(rlvr_ids) != rlvr_count:
        raise RuntimeError("Teacher partition size changed")

    canonical_path = layout.capacity / "verified_train_solutions.jsonl"
    canonical = _read_jsonl(canonical_path)
    by_id = {str(record["id"]): record for record in canonical}
    if len(by_id) != EXPECTED_TRAINING_TASKS or set(by_id) != set(ordered_ids):
        raise RuntimeError("canonical solutions do not match the 678 training tasks")
    warmstart_path = layout.run_root / "teacher_warmstart_solutions.jsonl"
    _write_jsonl(warmstart_path, [by_id[task_id] for task_id in sft_ids])

    source_gate_path = (root / str(settings["source_student_capacity_gate"])).resolve()
    source_gate = json.loads(source_gate_path.read_text(encoding="utf-8"))
    if (
        source_gate.get("status") != "passed"
        or int(source_gate.get("development_correct", 0)) < 52
    ):
        raise RuntimeError(f"source Mistral Student capacity did not pass: {source_gate}")
    source_s0 = (root / str(settings["source_student_base"])).resolve()
    if "qwen" in str(source_s0).lower() or not checkpoint_complete(source_s0):
        raise RuntimeError(f"invalid reusable Mistral Student S0: {source_s0}")
    layout.student_base.parent.mkdir(parents=True, exist_ok=True)
    if layout.student_base.is_symlink() or layout.student_base.exists():
        if layout.student_base.resolve() != source_s0:
            raise RuntimeError(
                f"v7 Student S0 already points elsewhere: {layout.student_base.resolve()}"
            )
    else:
        os.symlink(source_s0, layout.student_base, target_is_directory=True)

    manifest = {
        "status": "frozen",
        "protocol": V7_PROTOCOL,
        "seed": int(settings["split_seed"]),
        "source_training_tasks": EXPECTED_TRAINING_TASKS,
        "teacher_sft_count": len(sft_ids),
        "teacher_rlvr_count": len(rlvr_ids),
        "teacher_sft_ids": sft_ids,
        "teacher_rlvr_ids": rlvr_ids,
        "disjoint": not bool(set(sft_ids) & set(rlvr_ids)),
        "exact_union": set(sft_ids) | set(rlvr_ids) == set(ordered_ids),
        "teacher_warmstart_solutions": str(warmstart_path.resolve()),
        "student_capacity_gate": str(source_gate_path),
        "student_capacity_development_correct": int(
            source_gate["development_correct"]
        ),
        "student_base": str(source_s0),
        "frozen_test_accessed": False,
    }
    write_json(layout.run_root / "teacher_data_split.json", manifest)
    write_json(layout.state_root / "teacher_data_split.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    prepare_disjoint(args.root.resolve())


if __name__ == "__main__":
    main()
