from __future__ import annotations

import json
from pathlib import Path

from .io import atomic_json
from .minimal_routes import evaluate_items, evaluation_item, student_base_item, summary_for_item
from .minimal_runtime import (
    ARTIFACT_ROOT, DEVELOPMENT, TEACHER_ID, TEACHER_REVISION, TEACHER_ROUTE,
)


def main() -> None:
    evaluation_root = ARTIFACT_ROOT / "development"
    teacher_items = [
        evaluation_item("teacher_t0", TEACHER_ID, 0, evaluation_root, TEACHER_REVISION),
        evaluation_item("teacher_t50", TEACHER_ROUTE / "checkpoint-50", 50, evaluation_root),
        evaluation_item("teacher_t100", TEACHER_ROUTE / "checkpoint-100", 100, evaluation_root),
        evaluation_item("teacher_t200", TEACHER_ROUTE / "checkpoint-200", 200, evaluation_root),
        evaluation_item("teacher_t300", TEACHER_ROUTE / "checkpoint-300", 300, evaluation_root),
        evaluation_item("teacher_t400", TEACHER_ROUTE / "final_model", 400, evaluation_root),
    ]
    items = teacher_items + [student_base_item(evaluation_root)]
    for item in items:
        if not item.get("revision") and not Path(item["model"]).exists():
            raise FileNotFoundError(item["model"])
    evaluate_items(items, DEVELOPMENT, ARTIFACT_ROOT / "plans" / "teacher_and_student_base_dev.json")
    curve = [summary_for_item(item) for item in teacher_items]
    payload = {
        "teacher_curve": curve,
        "teacher_t0_correct": int(curve[0]["correct"]),
        "teacher_t400_correct": int(curve[-1]["correct"]),
        "student_s0": summary_for_item(items[-1]),
        "teacher_acquisition": int(curve[-1]["correct"]) > int(curve[0]["correct"]),
    }
    atomic_json(ARTIFACT_ROOT / "development" / "teacher_gate.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
