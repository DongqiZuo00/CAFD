from __future__ import annotations

import json
from pathlib import Path

from .minimal_routes import evaluate_items, evaluation_item, run
from .minimal_runtime import (
    ARTIFACT_ROOT, DEVELOPMENT, ROOT, RUN_ROOT, TEACHER_ID, TEACHER_REVISION,
    TEACHER_ROUTE, latest_checkpoint, load_summary,
)


CURRENT_PROBE = ROOT / "runs" / "qwen_olymmath_easy" / "teacher" / "lr_probes" / "5e-07"
ALTERNATE_PROBE = RUN_ROOT / "teacher_alternate_probe_1e-6"


def train(output: Path, learning_rate: float, steps: int, milestones: str, resume: Path | None = None) -> None:
    command = [
        "torchrun", "--standalone", "--nproc_per_node=4", "-m", "gap_validation.train_grpo",
        "--model-id", TEACHER_ID,
        "--revision", TEACHER_REVISION,
        "--dataset", str(ROOT / "data" / "frozen" / "math_training" / "teacher_prompt_stream.jsonl"),
        "--output-dir", str(output),
        "--learning-rate", str(learning_rate),
        "--max-steps", str(steps),
        "--max-completion-length", "8192",
        "--save-milestones", milestones,
        "--deepspeed", str(ROOT / "configs" / "gap_validation" / "deepspeed_zero3.json"),
        "--prompts-per-device", "2",
        "--gradient-accumulation-steps", "1",
    ]
    if resume is not None:
        command.extend(["--resume-from-checkpoint", str(resume)])
    run(command)


def ensure_probe_evaluation(root: Path, condition: str) -> dict:
    summary = root / "development_summary.json"
    if summary.exists():
        return load_summary(summary)
    final_model = root / "final_model"
    if not final_model.exists():
        raise FileNotFoundError(final_model)
    item = evaluation_item(condition, final_model, 50, root)
    item["predictions"] = str(root / "development_predictions.jsonl")
    item["summary"] = str(summary)
    evaluate_items([item], DEVELOPMENT, ARTIFACT_ROOT / "plans" / f"{condition}_recovery_eval.json")
    return load_summary(summary)


def main() -> None:
    selected_lr = 5e-7
    current_summary = CURRENT_PROBE / "development_summary.json"
    if current_summary.exists() or (CURRENT_PROBE / "final_model").exists():
        probe = ensure_probe_evaluation(CURRENT_PROBE, "teacher_probe_5e-7")
        if not 0.0 <= float(probe["exact_answer_accuracy"]) <= 1.0:
            raise RuntimeError("current Teacher probe produced an invalid development score")
    else:
        selected_lr = 1e-6
        if not (ALTERNATE_PROBE / "final_model").exists():
            train(ALTERNATE_PROBE, selected_lr, 50, "50", latest_checkpoint(ALTERNATE_PROBE))
        probe = ensure_probe_evaluation(ALTERNATE_PROBE, "teacher_alternate_probe_1e-6")
        if not 0.0 <= float(probe["exact_answer_accuracy"]) <= 1.0:
            raise RuntimeError("the single permitted alternate Teacher probe is invalid")

    if not (TEACHER_ROUTE / "run_summary.json").exists():
        train(
            TEACHER_ROUTE,
            selected_lr,
            400,
            "50,100,200,300,400",
            latest_checkpoint(TEACHER_ROUTE),
        )
    payload = {
        "state": "COMPLETED",
        "selected_learning_rate": selected_lr,
        "formal_route": str(TEACHER_ROUTE),
        "formal_updates": int(load_summary(TEACHER_ROUTE / "run_summary.json")["global_step"]),
    }
    (RUN_ROOT / "teacher_recovery.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
