from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from .io import ARTIFACT_ROOT, atomic_json


STUDENT_ID = "Qwen/Qwen3.5-2B"
STUDENT_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
METHODS = {
    "teacher_rft": {"steps": 200, "milestones": (25, 50, 100, 150, 200)},
    "capacity_control": {"steps": 800, "milestones": (100, 200, 400, 600, 800)},
}


def run(command: list[str]) -> None:
    completed = subprocess.run(command, env={**os.environ, "PYTHONHASHSEED": "42"}, check=False)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {command}")


def train_command(dataset: Path, output: Path, learning_rate: float, steps: int, milestones: str) -> list[str]:
    return [
        "python", "-m", "gap_validation.train_sft",
        "--model-id", STUDENT_ID,
        "--revision", STUDENT_REVISION,
        "--dataset", str(dataset),
        "--output-dir", str(output),
        "--learning-rate", str(learning_rate),
        "--max-steps", str(steps),
        "--save-milestones", milestones,
    ]


def evaluate(model: str | Path, dataset: Path, predictions: Path, summary: Path, revision: str | None = None) -> float:
    command = [
        "python", "-m", "gap_validation.evaluate_math",
        "--model", str(model),
        "--dataset", str(dataset),
        "--output", str(predictions),
        "--summary-output", str(summary),
        "--max-new-tokens", "8192",
    ]
    if revision:
        command.extend(["--revision", revision])
    run(command)
    return float(json.loads(summary.read_text(encoding="utf-8"))["exact_answer_accuracy"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    development = Path("data/frozen/olymmath/development.jsonl")
    final_test = Path("data/frozen/olymmath/final_test.jsonl")

    probe_scores: dict[str, float] = {}
    for learning_rate in (1e-6, 2e-6, 5e-6):
        key = f"{learning_rate:.0e}"
        probe = args.root / "lr_probes" / key
        summary = probe / "development_summary.json"
        if not summary.exists():
            run(train_command(args.dataset, probe, learning_rate, 50, "50"))
            probe_scores[key] = evaluate(
                probe / "final_model", development, probe / "development_predictions.jsonl", summary
            )
        else:
            probe_scores[key] = float(json.loads(summary.read_text(encoding="utf-8"))["exact_answer_accuracy"])
    best_score = max(probe_scores.values())
    selected_lr = min(float(key) for key, value in probe_scores.items() if value == best_score)
    atomic_json(
        args.root / "learning_rate_selection.json",
        {
            "selection_split": "olymmath_english_easy_development",
            "metric": "exact_answer_accuracy",
            "candidate_scores": probe_scores,
            "tie_break": "lower_learning_rate",
            "selected_learning_rate": selected_lr,
        },
    )

    protocol = METHODS[args.method]
    full = args.root / "selected_full_budget"
    if not (full / "run_summary.json").exists():
        run(
            train_command(
                args.dataset,
                full,
                selected_lr,
                protocol["steps"],
                ",".join(str(step) for step in protocol["milestones"]),
            )
        )
    candidates: dict[int, tuple[str | Path, str | None]] = {0: (STUDENT_ID, STUDENT_REVISION)}
    for step in protocol["milestones"]:
        candidates[step] = (full / ("final_model" if step == protocol["steps"] else f"checkpoint-{step}"), None)
    development_scores: dict[str, float] = {}
    for step, (model, revision) in candidates.items():
        evaluation = args.root / "development_selection" / f"step-{step}"
        summary = evaluation / "summary.json"
        score = (
            float(json.loads(summary.read_text(encoding="utf-8"))["exact_answer_accuracy"])
            if summary.exists()
            else evaluate(model, development, evaluation / "predictions.jsonl", summary, revision)
        )
        development_scores[str(step)] = score
    selected_step = min(
        int(step) for step, score in development_scores.items() if score == max(development_scores.values())
    )
    selected_model, selected_revision = candidates[selected_step]
    selection = {
        "method": args.method,
        "development_scores": development_scores,
        "tie_break": "earlier_optimizer_step",
        "selected_step": selected_step,
        "selected_model": str(selected_model),
    }
    atomic_json(args.root / "checkpoint_selection.json", selection)
    final_summary = args.root / "final_test_summary.json"
    if not final_summary.exists():
        evaluate(
            selected_model,
            final_test,
            ARTIFACT_ROOT / "raw_predictions" / f"qwen_olymmath_easy_{args.method}.jsonl",
            final_summary,
            selected_revision,
        )
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
