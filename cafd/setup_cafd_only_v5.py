"""Materialize immutable CAFD-only v5 inputs without copying model weights."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .configuration import load_experiment_config
from .layout import experiment_layout
from .training_common import checkpoint_complete, write_json


RUN_ID = "cafd_only_v5_full_route"


def _require_checkpoint(path: Path) -> str:
    path = path.resolve()
    if not checkpoint_complete(path):
        raise RuntimeError(f"incomplete v5 checkpoint: {path}")
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = load_experiment_config(root)
    if config["experiment"]["run_id"] != RUN_ID:
        raise RuntimeError("wrong CAFD-only v5 run_id")
    layout = experiment_layout(root, config)

    source_student = (
        root / "runs/cafd/experiments/chat_hier_v3/student_base/S0"
    ).resolve()
    if not checkpoint_complete(source_student):
        raise RuntimeError(f"incomplete Student S0: {source_student}")
    target_student = layout.student_base
    target_student.parent.mkdir(parents=True, exist_ok=True)
    if target_student.is_symlink() or target_student.exists():
        if target_student.resolve() != source_student:
            raise RuntimeError("v5 Student S0 points to the wrong checkpoint")
    else:
        os.symlink(source_student, target_student, target_is_directory=True)

    teacher_revision = str(config["models"]["teacher"]["revision"])
    raw_teacher = (
        root
        / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507"
        / "snapshots"
        / teacher_revision
    )
    checkpoints = [
        _require_checkpoint(raw_teacher),
        _require_checkpoint(
            root / "runs/cafd/experiments/teacher_capacity_v1/teacher_capacity/SFT25"
        ),
        _require_checkpoint(
            root / "runs/cafd/experiments/chat_hier_v4b_sft25/teacher/candidates/T20"
        ),
        _require_checkpoint(
            root / "runs/cafd/experiments/chat_hier_v4b_sft25/teacher/candidates/T40"
        ),
        _require_checkpoint(
            root / "runs/cafd/experiments/chat_hier_v4b_sft25/teacher/candidates/T60"
        ),
        _require_checkpoint(
            root / "runs/cafd/experiments/chat_hier_v4b_sft25/teacher/candidates/T100"
        ),
    ]
    steps = ["raw", "sft25", "rl20", "rl40", "rl60", "rl100"]
    route = {
        "status": "frozen",
        "kind": "continuous_full_acquisition",
        "checkpoints": [
            {"step": step, "checkpoint": checkpoint}
            for step, checkpoint in zip(steps, checkpoints, strict=True)
        ],
    }
    layout.teacher.mkdir(parents=True, exist_ok=True)
    write_json(layout.teacher / "route.json", route)
    write_json(
        layout.state_root / "setup.json",
        {
            "stage": "inputs_frozen",
            "run_id": RUN_ID,
            "student_base": str(target_student.resolve()),
            "teacher_route": route,
        },
    )


if __name__ == "__main__":
    main()
