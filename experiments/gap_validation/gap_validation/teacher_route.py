from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from .io import atomic_json


QWEN_TEACHER = "Qwen/Qwen3.5-9B"
QWEN_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {command}")


def train_command(output: Path, learning_rate: float, steps: int, milestones: str) -> list[str]:
    return [
        "torchrun", "--standalone", "--nproc_per_node=4", "-m", "gap_validation.train_grpo",
        "--model-id", QWEN_TEACHER,
        "--revision", QWEN_REVISION,
        "--dataset", "data/frozen/math_training/teacher_prompt_stream.jsonl",
        "--output-dir", str(output),
        "--learning-rate", str(learning_rate),
        "--max-steps", str(steps),
        "--max-completion-length", "8192",
        "--save-milestones", milestones,
        "--deepspeed", "configs/gap_validation/deepspeed_zero3.json",
        "--prompts-per-device", "2",
        "--gradient-accumulation-steps", "1",
    ]


def evaluate(model: Path, prediction_path: Path, summary_path: Path) -> float:
    run(
        [
            "python", "-m", "gap_validation.evaluate_math",
            "--model", str(model),
            "--dataset", "data/frozen/olymmath/development.jsonl",
            "--output", str(prediction_path),
            "--summary-output", str(summary_path),
            "--max-new-tokens", "8192",
        ]
    )
    return float(json.loads(summary_path.read_text(encoding="utf-8"))["exact_answer_accuracy"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/qwen_olymmath_easy/teacher"))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "42"
    candidate_scores: dict[str, float] = {}
    for learning_rate in (5e-7, 1e-6, 2e-6):
        key = f"{learning_rate:.0e}"
        probe = args.root / "lr_probes" / key
        summary = probe / "development_summary.json"
        if not summary.exists():
            run(train_command(probe, learning_rate, 50, "50"), env=env)
            candidate_scores[key] = evaluate(
                probe / "final_model",
                probe / "development_predictions.jsonl",
                summary,
            )
        else:
            candidate_scores[key] = float(json.loads(summary.read_text(encoding="utf-8"))["exact_answer_accuracy"])
    selected = min(
        (float(key), score) for key, score in candidate_scores.items()
        if score == max(candidate_scores.values())
    )[0]
    selection = {
        "selection_split": "olymmath_english_easy_development",
        "metric": "exact_answer_accuracy",
        "candidate_scores": candidate_scores,
        "tie_break": "lower_learning_rate",
        "selected_learning_rate": selected,
    }
    atomic_json(args.root / "learning_rate_selection.json", selection)
    final = args.root / "selected_full_route"
    if not (final / "run_summary.json").exists():
        run(train_command(final, selected, 400, "50,100,200,300,400"), env=env)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
