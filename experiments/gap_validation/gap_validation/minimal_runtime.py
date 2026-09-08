from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from .io import ROOT, atomic_json


RUN_ID = "qwen_olymmath_easy_minimal_gap_seed42_20260826"
RUN_ROOT = ROOT / "runs" / RUN_ID
ARTIFACT_ROOT = ROOT / "artifacts" / "gap_validation" / "minimal" / RUN_ID
DEVELOPMENT = ROOT / "data" / "frozen" / "olymmath" / "development.jsonl"
FINAL_TEST = ROOT / "data" / "frozen" / "olymmath" / "final_test.jsonl"
STUDENT_STREAM = ROOT / "data" / "frozen" / "math_training" / "student_prompt_stream.jsonl"
TEACHER_ROUTE = ROOT / "runs" / "qwen_olymmath_easy" / "teacher" / "selected_full_route"

TEACHER_ID = "Qwen/Qwen3.5-9B"
TEACHER_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
STUDENT_ID = "Qwen/Qwen3.5-2B"
STUDENT_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"


def completion_tokens_from_history(log_history: list[dict[str, Any]], slots_per_update: int = 32) -> int:
    return round(
        sum(
            float(row["completions/mean_length"]) * slots_per_update
            for row in log_history
            if row.get("completions/mean_length") is not None
        )
    )


def generation_events_from_history(log_history: list[dict[str, Any]]) -> int:
    return sum(row.get("completions/mean_length") is not None for row in log_history)


def distributed_max(value: float) -> float:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=torch.cuda.current_device())
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return float(tensor.item())


def peak_gpu_memory_bytes() -> int:
    local = float(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0.0
    return int(distributed_max(local))


def elapsed_seconds(started: float) -> float:
    return distributed_max(time.monotonic() - started)


def latest_checkpoint(root: Path) -> Path | None:
    checkpoints = []
    for path in root.glob("checkpoint-*"):
        if path.is_dir():
            try:
                checkpoints.append((int(path.name.rsplit("-", 1)[1]), path))
            except ValueError:
                continue
    return max(checkpoints, default=(0, None))[1]


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"summary must be an object: {path}")
    return value


def write_run_status(stage: str, state: str, **extra: Any) -> None:
    path = ARTIFACT_ROOT / "status.json"
    existing = load_summary(path) if path.exists() else {"run_id": RUN_ID, "history": []}
    existing["stage"] = stage
    existing["state"] = state
    existing["updated_at"] = time.time()
    existing["history"].append({"stage": stage, "state": state, "time": time.time(), **extra})
    existing.update(extra)
    atomic_json(path, existing)
