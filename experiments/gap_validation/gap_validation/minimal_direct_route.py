from __future__ import annotations

import json

from .io import atomic_json
from .minimal_routes import (
    direct_root, evaluate_items, evaluation_item, run, select_checkpoint, student_base_item,
)
from .minimal_runtime import ARTIFACT_ROOT, DEVELOPMENT, ROOT, STUDENT_ID, STUDENT_REVISION, STUDENT_STREAM, latest_checkpoint


def main() -> None:
    root = direct_root()
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "run_summary.json").exists():
        command = [
            "torchrun", "--standalone", "--nproc_per_node=4", "-m", "gap_validation.minimal_train_grpo",
            "--model-id", STUDENT_ID,
            "--revision", STUDENT_REVISION,
            "--dataset", str(STUDENT_STREAM),
            "--output-dir", str(root),
            "--learning-rate", "1e-6",
            "--max-steps", "200",
            "--save-milestones", "50,100,200",
            "--deepspeed", str(ROOT / "configs" / "gap_validation" / "deepspeed_zero3.json"),
        ]
        checkpoint = latest_checkpoint(root)
        if checkpoint is not None:
            command.extend(["--resume-from-checkpoint", str(checkpoint)])
        run(command)
    evaluation_root = ARTIFACT_ROOT / "development"
    items = [
        student_base_item(evaluation_root),
        evaluation_item("direct_step_50", root / "checkpoint-50", 50, evaluation_root),
        evaluation_item("direct_step_100", root / "checkpoint-100", 100, evaluation_root),
        evaluation_item("direct_step_200", root / "final_model", 200, evaluation_root),
    ]
    evaluate_items(items, DEVELOPMENT, ARTIFACT_ROOT / "plans" / "direct_dev.json")
    selection = {"method": "student_direct_rlvr", "learning_rate": 1e-6, **select_checkpoint(items)}
    atomic_json(root / "checkpoint_selection.json", selection)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
