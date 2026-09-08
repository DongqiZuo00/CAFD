"""Freeze all development-selected checkpoints before official-test access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .configuration import load_experiment_config
from .layout import experiment_layout
from .training_common import write_json


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = load_experiment_config(root)
    layout = experiment_layout(root, config)
    student_base = layout.student_base
    teacher_gate = _load(layout.teacher / "gate.json")
    capacity_gate = _load(layout.capacity / "gate.json")
    if teacher_gate.get("status") != "passed":
        raise RuntimeError("TEACHER_ROUTE_FAILED")
    if capacity_gate.get("status") != "passed":
        raise RuntimeError("CAPACITY_BLOCKED")
    teacher_route = _load(layout.teacher / "route.json")["checkpoints"]
    if len(teacher_route) != 6:
        raise RuntimeError("Teacher route does not contain six frozen checkpoints")
    selected = {method: _load(layout.method(method) / "selected.json") for method in ("direct", "endpoint", "progressive", "cafd")}
    frozen = {
        "seed": 2027,
        "metric": "official held-out full-pass/pass@1",
        "conditions": {
            "teacher_base": {"label": "Teacher Base", "checkpoint": teacher_route[0]["checkpoint"], "selected_step": 0, "budget": 0},
            "teacher_rlvr": {"label": "Teacher RLVR", "checkpoint": teacher_gate["checkpoint"], "selected_step": teacher_gate["selected_step"], "budget": teacher_gate["actual_updates"], **{k: teacher_gate[k] for k in ("generated_tokens", "gpu_hours")}},
            "student_base": {"label": "Student Base", "checkpoint": str(student_base.resolve()), "selected_step": 0, "budget": 0},
            "capacity": {"label": "Capacity Control", "checkpoint": capacity_gate["checkpoint"], "selected_step": capacity_gate["selected_step"], "budget": capacity_gate["actual_updates"], **{k: capacity_gate[k] for k in ("generated_tokens", "gpu_hours")}},
            "direct": {"label": "Direct RLVR", "budget": 200, **selected["direct"]},
            "endpoint": {"label": "Endpoint OPD", "budget": 200, **selected["endpoint"]},
            "progressive": {"label": "Progressive GKD", "budget": 200, **selected["progressive"]},
            "cafd": {"label": "CAFD-v1", "budget": 200, **selected["cafd"]},
        },
    }
    for condition in frozen["conditions"].values():
        if not Path(condition["checkpoint"]).is_dir():
            raise FileNotFoundError(condition["checkpoint"])
        condition.setdefault("generated_tokens", 0)
        condition.setdefault("teacher_scored_tokens", 0)
        condition.setdefault("gpu_hours", 0.0)
    write_json(layout.frozen_selection, frozen)
    write_json(layout.state_root / "freeze.json", {"stage": "selection_frozen", "conditions": list(frozen["conditions"])})


if __name__ == "__main__":
    main()
