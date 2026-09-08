"""Freeze and summarize the single support-corrected CAFD condition."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from .configuration import load_experiment_config
from .layout import experiment_layout
from .training_common import write_json


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def freeze(root: Path) -> None:
    config = load_experiment_config(root)
    layout = experiment_layout(root, config)
    selected = _load(layout.method("cafd") / "selected.json")
    if int(config["student"]["updates"]) != 200:
        raise RuntimeError("refusing to freeze a non-200-update CAFD run")
    if selected.get("condition") != "CAFD-v1":
        raise RuntimeError("unexpected CAFD selection record")
    checkpoint = Path(str(selected["checkpoint"]))
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    if int(selected["selected_step"]) not in {
        int(step) for step in config["student"]["milestones"]
    }:
        raise RuntimeError("selected checkpoint was not a development milestone")
    frozen = {
        "seed": int(config["seed"]),
        "metric": "official held-out full-pass/pass@1",
        "selection_split": "development",
        "protocol": str(config["protocol"]),
        "conditions": {
            "cafd": {
                "label": "CAFD-v1 (support-corrected replay)",
                "budget": 200,
                **selected,
            }
        },
    }
    if layout.frozen_selection.exists():
        if _load(layout.frozen_selection) != frozen:
            raise RuntimeError("frozen CAFD selection already exists with different content")
        return
    write_json(layout.frozen_selection, frozen)
    write_json(
        layout.state_root / "freeze.json",
        {"stage": "selection_frozen", "conditions": ["cafd"]},
    )


def _wilson(correct: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = correct / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _write_csv(path: Path, row: dict[str, Any], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def finalize(root: Path) -> None:
    config = load_experiment_config(root)
    layout = experiment_layout(root, config)
    record = _load(layout.final / "cafd.json")
    correct = int(record["correct"])
    total = int(record["total"])
    if total != int(config["benchmark"]["held_out_size"]):
        raise RuntimeError(f"unexpected held-out size: {total}")
    low, high = _wilson(correct, total)
    score = {
        "condition": record["label"],
        "correct": correct,
        "total": total,
        "test_full_pass": float(record["accuracy"]),
        "ci95_low": low,
        "ci95_high": high,
        "development_accuracy": float(record["development_accuracy"]),
        "selected_step": int(record["selected_step"]),
        "budget_updates": int(record["budget"]),
        "checkpoint": str(record["checkpoint"]),
    }
    cost = {
        "condition": record["label"],
        "logical_generated_tokens": int(record["logical_generated_tokens"]),
        "logical_teacher_scored_tokens": int(record["logical_teacher_scored_tokens"]),
        "actual_generated_tokens": int(record["actual_generated_tokens"]),
        "actual_teacher_scored_tokens": int(record["actual_teacher_scored_tokens"]),
        "evaluation_generated_tokens": int(record["evaluation_generated_tokens"]),
        "gpu_hours": float(record["gpu_hours"]),
    }
    _write_csv(
        layout.artifact_root / "final_scores.csv",
        score,
        list(score),
    )
    _write_csv(
        layout.artifact_root / "costs.csv",
        cost,
        list(cost),
    )
    write_json(
        layout.artifact_root / "final_summary.json",
        {"score": score, "cost": cost},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["freeze", "finalize"], required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "freeze":
        freeze(root)
    else:
        finalize(root)


if __name__ == "__main__":
    main()
