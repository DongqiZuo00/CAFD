from __future__ import annotations

import json
from pathlib import Path

from .io import atomic_json
from .minimal_routes import (
    evaluate_items, evaluation_item, oracle_root, run, select_checkpoint, student_base_item,
)
from .minimal_runtime import (
    ARTIFACT_ROOT, DEVELOPMENT, STUDENT_ID, STUDENT_REVISION, STUDENT_STREAM,
    latest_checkpoint, load_summary,
)


def train_to(root: Path, dataset: Path, target: int) -> None:
    if (root / f"run_summary_step_{target}.json").exists():
        return
    command = [
        "torchrun", "--standalone", "--nproc_per_node=4", "-m", "gap_validation.minimal_train_oracle",
        "--model-id", STUDENT_ID,
        "--revision", STUDENT_REVISION,
        "--dataset", str(dataset),
        "--output-dir", str(root),
        "--learning-rate", "2e-6",
        "--budget-cap", "800",
        "--stop-after", str(target),
        "--save-milestones", "100,200,400,800",
    ]
    checkpoint = latest_checkpoint(root)
    if checkpoint is not None:
        command.extend(["--resume-from-checkpoint", str(checkpoint)])
    run(command)


def evaluate_milestones(root: Path, steps: list[int], items: list[dict]) -> list[dict]:
    evaluation_root = ARTIFACT_ROOT / "development"
    new_items = [
        evaluation_item(f"oracle_step_{step}", root / f"checkpoint-{step}", step, evaluation_root)
        for step in steps
    ]
    evaluate_items(
        new_items,
        DEVELOPMENT,
        ARTIFACT_ROOT / "plans" / f"oracle_steps_{'_'.join(map(str, steps))}_dev.json",
    )
    items.extend(new_items)
    return [load_summary(Path(item["summary"])) for item in new_items]


def main() -> None:
    root = oracle_root()
    root.mkdir(parents=True, exist_ok=True)
    data_root = root / "data"
    data_status = data_root / "status.json"
    if not data_status.exists():
        run(
            [
                "python", "-m", "gap_validation.minimal_prepare_oracle",
                "--input", str(STUDENT_STREAM),
                "--output", str(data_root / "oracle_stream.jsonl"),
                "--status", str(data_status),
                "--failure-output", str(data_root / "canonical_failures.jsonl"),
                "--max-updates", "800",
                "--sequences-per-update", "32",
            ]
        )
    status = load_summary(data_status)
    if status["state"] != "COMPLETED":
        raise RuntimeError("verified canonical Oracle data are incomplete")

    evaluation_root = ARTIFACT_ROOT / "development"
    items: list[dict] = []
    train_to(root, data_root / "oracle_stream.jsonl", 200)
    score100, score200 = evaluate_milestones(root, [100, 200], items)
    teacher = load_summary(evaluation_root / "teacher_t400" / "summary.json")
    teacher_correct = int(teacher["correct"])
    last_score = score200
    if int(score200["correct"]) < teacher_correct and int(score200["correct"]) > int(score100["correct"]):
        train_to(root, data_root / "oracle_stream.jsonl", 400)
        score400 = evaluate_milestones(root, [400], items)[0]
        last_score = score400
        if int(score400["correct"]) < teacher_correct and int(score400["correct"]) > int(score200["correct"]):
            train_to(root, data_root / "oracle_stream.jsonl", 800)
            last_score = evaluate_milestones(root, [800], items)[0]

    selection = {
        "method": "student_oracle_supervision",
        "learning_rate": 2e-6,
        **select_checkpoint(items),
    }
    selection["capacity_established"] = (
        int(selection["development_correct"]) >= teacher_correct
        and int(selection["development_correct"])
        > int(load_summary(evaluation_root / "student_s0" / "summary.json")["correct"])
    )
    selection["last_evaluated_step"] = int(last_score["checkpoint_step"])
    selection["canonical_data_status"] = status
    atomic_json(root / "checkpoint_selection.json", selection)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
