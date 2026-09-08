from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .io import atomic_json
from .minimal_runtime import RUN_ROOT, STUDENT_ID, STUDENT_REVISION, load_summary


def run(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        env={**os.environ, "PYTHONHASHSEED": "42"},
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {command}")


def evaluation_item(
    condition: str,
    model: str | Path,
    checkpoint_step: int,
    evaluation_root: Path,
    revision: str | None = None,
) -> dict[str, Any]:
    root = evaluation_root / condition
    return {
        "condition": condition,
        "model": str(model),
        "revision": revision,
        "checkpoint_step": checkpoint_step,
        "predictions": str(root / "predictions.jsonl"),
        "summary": str(root / "summary.json"),
        "max_new_tokens": 8192,
    }


def evaluate_items(items: list[dict[str, Any]], dataset: Path, plan_path: Path) -> None:
    plan = {"dataset": str(dataset), "seed": 42, "items": items}
    atomic_json(plan_path, plan)
    run(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node=4",
            "-m",
            "gap_validation.minimal_evaluate_math",
            "--plan",
            str(plan_path),
        ]
    )


def summary_for_item(item: dict[str, Any]) -> dict[str, Any]:
    return load_summary(Path(item["summary"]))


def select_checkpoint(items: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [(item, summary_for_item(item)) for item in items]
    best_correct = max(int(summary["correct"]) for _, summary in scored)
    selected_item, selected_summary = min(
        (
            (item, summary)
            for item, summary in scored
            if int(summary["correct"]) == best_correct
        ),
        key=lambda pair: int(pair[0]["checkpoint_step"]),
    )
    return {
        "selected_step": int(selected_item["checkpoint_step"]),
        "selected_model": selected_item["model"],
        "selected_revision": selected_item.get("revision"),
        "development_correct": best_correct,
        "development_total": int(selected_summary["count"]),
        "development_accuracy": float(selected_summary["exact_answer_accuracy"]),
        "tie_break": "earlier_optimizer_step",
        "candidate_summaries": {
            str(item["checkpoint_step"]): summary for item, summary in scored
        },
    }


def student_base_item(evaluation_root: Path) -> dict[str, Any]:
    return evaluation_item(
        "student_s0",
        STUDENT_ID,
        0,
        evaluation_root,
        revision=STUDENT_REVISION,
    )


def direct_root() -> Path:
    return RUN_ROOT / "student_direct_rlvr"


def oracle_root() -> Path:
    return RUN_ROOT / "student_oracle"


def opd_root() -> Path:
    return RUN_ROOT / "student_final_opd"
