from __future__ import annotations

import json

from .io import atomic_json
from .minimal_routes import (
    evaluate_items, evaluation_item, opd_root, run, select_checkpoint, student_base_item,
)
from .minimal_runtime import (
    ARTIFACT_ROOT, DEVELOPMENT, STUDENT_ID, STUDENT_REVISION, STUDENT_STREAM,
    TEACHER_ROUTE, latest_checkpoint,
)


def main() -> None:
    root = opd_root()
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "run_summary.json").exists():
        command = [
            "torchrun", "--standalone", "--nproc_per_node=4", "-m", "gap_validation.minimal_train_opd",
            "--student-model", STUDENT_ID,
            "--student-revision", STUDENT_REVISION,
            "--teacher", str(TEACHER_ROUTE / "final_model"),
            "--dataset", str(STUDENT_STREAM),
            "--output-dir", str(root),
            "--learning-rate", "2e-6",
            "--max-steps", "200",
            "--save-milestones", "50,100,200",
        ]
        checkpoint = latest_checkpoint(root)
        if checkpoint is not None:
            command.extend(["--resume-from-checkpoint", str(checkpoint)])
        run(command)
    evaluation_root = ARTIFACT_ROOT / "development"
    items = [
        student_base_item(evaluation_root),
        evaluation_item("opd_step_50", root / "checkpoint-50", 50, evaluation_root),
        evaluation_item("opd_step_100", root / "checkpoint-100", 100, evaluation_root),
        evaluation_item("opd_step_200", root / "final_model", 200, evaluation_root),
    ]
    evaluate_items(items, DEVELOPMENT, ARTIFACT_ROOT / "plans" / "opd_dev.json")
    selection = {"method": "student_final_opd", "learning_rate": 2e-6, **select_checkpoint(items)}
    atomic_json(root / "checkpoint_selection.json", selection)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
