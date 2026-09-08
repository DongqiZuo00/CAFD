from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .io import atomic_json
from .minimal_gates import development_gates
from .minimal_routes import evaluate_items, evaluation_item
from .minimal_runtime import (
    ARTIFACT_ROOT, FINAL_TEST, ROOT, RUN_ID, STUDENT_ID, STUDENT_REVISION, STUDENT_STREAM,
    TEACHER_ID, TEACHER_REVISION, TEACHER_ROUTE, completion_tokens_from_history,
    generation_events_from_history,
    load_summary,
)
from .splits import load_jsonl


CONDITIONS = [
    "teacher_t0",
    "teacher_t400",
    "student_s0",
    "student_oracle",
    "student_direct_rlvr",
    "student_final_opd",
]
LABELS = {
    "teacher_t0": "Teacher Base T0",
    "teacher_t400": "Teacher RLVR T400",
    "student_s0": "Student Base S0",
    "student_oracle": "Student Oracle",
    "student_direct_rlvr": "Student Direct RLVR",
    "student_final_opd": "Student Final OPD",
}


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def paired_bootstrap(correctness: dict[str, np.ndarray], replicates: int = 10000) -> dict[str, Any]:
    size = len(correctness["teacher_t0"])
    if any(len(values) != size for values in correctness.values()):
        raise ValueError("all conditions must share the same item count")
    rng = np.random.default_rng(42)
    indices = rng.integers(0, size, size=(replicates, size))
    means = {name: values[indices].mean(axis=1) for name, values in correctness.items()}
    delta_t = means["teacher_t400"] - means["teacher_t0"]
    delta_o = means["student_oracle"] - means["student_s0"]
    gap = np.minimum(means["teacher_t400"], means["student_oracle"]) - np.maximum(
        means["student_direct_rlvr"], means["student_final_opd"]
    )
    distributions = {"Delta_T": delta_t, "Delta_O": delta_o, "G": gap}
    return {
        "replicates": replicates,
        "seed": 42,
        "shared_resample_indices": True,
        "simultaneous_method": "Bonferroni one-sided 95 percent",
        "lower_percentile": 1.67,
        "lower_bounds": {
            name: float(np.percentile(values, 1.67)) for name, values in distributions.items()
        },
    }


def load_aligned_correctness(items: list[dict[str, Any]]) -> tuple[list[str], dict[str, np.ndarray]]:
    ordered_ids: list[str] | None = None
    values: dict[str, np.ndarray] = {}
    for item in items:
        rows = load_jsonl(Path(item["predictions"]))
        by_id = {str(row["prompt_id"]): bool(row["correct"]) for row in rows}
        if len(by_id) != len(rows):
            raise ValueError(f"duplicate prompt ID for {item['condition']}")
        if ordered_ids is None:
            ordered_ids = [str(row["prompt_id"]) for row in rows]
        if set(by_id) != set(ordered_ids):
            raise ValueError(f"prompt IDs do not align for {item['condition']}")
        values[item["condition"]] = np.asarray([by_id[prompt_id] for prompt_id in ordered_ids], dtype=float)
    assert ordered_ids is not None
    return ordered_ids, values


def trainer_cost(root: Path, world_size: int) -> dict[str, Any]:
    state_path = root / "trainer_state.json"
    if not state_path.exists():
        return {"generated_tokens": 0, "wall_time_seconds": 0.0, "gpu_hours": 0.0}
    state = load_summary(state_path)
    history = state.get("log_history", [])
    runtime = next(
        (float(row["train_runtime"]) for row in reversed(history) if row.get("train_runtime") is not None),
        0.0,
    )
    return {
        "generated_tokens": completion_tokens_from_history(history),
        "verifier_calls": generation_events_from_history(history) * 32,
        "wall_time_seconds": runtime,
        "gpu_hours": runtime * world_size / 3600,
    }


def teacher_observed_peak_memory() -> int | None:
    path = ARTIFACT_ROOT / "teacher_resource_monitor.json"
    if not path.exists():
        return None
    value = load_summary(path).get("observed_peak_gpu_memory_bytes")
    return int(value) if value is not None else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_final_items() -> list[dict[str, Any]]:
    development = ARTIFACT_ROOT / "development"
    direct = load_summary(
        Path("runs") / RUN_ID / "student_direct_rlvr" / "checkpoint_selection.json"
    )
    oracle = load_summary(Path("runs") / RUN_ID / "student_oracle" / "checkpoint_selection.json")
    opd = load_summary(Path("runs") / RUN_ID / "student_final_opd" / "checkpoint_selection.json")
    t0 = load_summary(development / "teacher_t0" / "summary.json")
    t400 = load_summary(development / "teacher_t400" / "summary.json")
    s0 = load_summary(development / "student_s0" / "summary.json")
    gates = development_gates(
        t0=int(t0["correct"]),
        t400=int(t400["correct"]),
        s0=int(s0["correct"]),
        direct=int(direct["development_correct"]),
        oracle=int(oracle["development_correct"]),
        opd=int(opd["development_correct"]),
    )
    if not gates["final_open"]:
        raise RuntimeError(f"frozen final gate is closed: {gates}")
    final_root = ARTIFACT_ROOT / "final"
    return [
        evaluation_item("teacher_t0", TEACHER_ID, 0, final_root, TEACHER_REVISION),
        evaluation_item("teacher_t400", TEACHER_ROUTE / "final_model", 400, final_root),
        evaluation_item("student_s0", STUDENT_ID, 0, final_root, STUDENT_REVISION),
        evaluation_item(
            "student_oracle", oracle["selected_model"], int(oracle["selected_step"]), final_root,
            oracle.get("selected_revision"),
        ),
        evaluation_item(
            "student_direct_rlvr", direct["selected_model"], int(direct["selected_step"]), final_root,
            direct.get("selected_revision"),
        ),
        evaluation_item(
            "student_final_opd", opd["selected_model"], int(opd["selected_step"]), final_root,
            opd.get("selected_revision"),
        ),
    ]


def cost_rows(items: list[dict[str, Any]], summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    run_root = Path("runs") / RUN_ID
    teacher_cost = trainer_cost(TEACHER_ROUTE, 4)
    direct_summary = load_summary(run_root / "student_direct_rlvr" / "run_summary.json")
    opd_summary = load_summary(run_root / "student_final_opd" / "run_summary.json")
    oracle_selection = load_summary(run_root / "student_oracle" / "checkpoint_selection.json")
    oracle_segments = [
        load_summary(path)
        for path in sorted((run_root / "student_oracle").glob("run_summary_step_*.json"))
    ]
    training = {
        "teacher_t0": {"updates": 0, "generated_tokens": 0, "teacher_scored_tokens": 0, "verifier_calls": 0, "wall_time_seconds": 0.0, "gpu_hours": 0.0, "peak_gpu_memory_bytes": None},
        "teacher_t400": {"updates": 400, "teacher_scored_tokens": 0, "peak_gpu_memory_bytes": teacher_observed_peak_memory(), **teacher_cost},
        "student_s0": {"updates": 0, "generated_tokens": 0, "teacher_scored_tokens": 0, "verifier_calls": 0, "wall_time_seconds": 0.0, "gpu_hours": 0.0, "peak_gpu_memory_bytes": None},
        "student_oracle": {
            "updates": int(oracle_selection["last_evaluated_step"]),
            "generated_tokens": 0,
            "teacher_scored_tokens": 0,
            "verifier_calls": int(oracle_selection["canonical_data_status"]["training_prompts"]),
            "wall_time_seconds": sum(float(row["wall_time_seconds_this_segment"]) for row in oracle_segments),
            "gpu_hours": sum(float(row["wall_time_seconds_this_segment"]) * 4 / 3600 for row in oracle_segments),
            "peak_gpu_memory_bytes": max((int(row["peak_gpu_memory_bytes"]) for row in oracle_segments), default=None),
        },
        "student_direct_rlvr": {
            "updates": 200,
            "generated_tokens": int(direct_summary["generated_tokens"]),
            "teacher_scored_tokens": 0,
            "verifier_calls": int(direct_summary["verifier_calls"]),
            "wall_time_seconds": float(direct_summary["wall_time_seconds"]),
            "gpu_hours": float(direct_summary["wall_time_seconds"]) * 4 / 3600,
            "peak_gpu_memory_bytes": int(direct_summary["peak_gpu_memory_bytes"]),
        },
        "student_final_opd": {
            "updates": 200,
            "generated_tokens": int(opd_summary["generated_tokens"]),
            "teacher_scored_tokens": int(opd_summary["teacher_scored_tokens"]),
            "verifier_calls": 0,
            "wall_time_seconds": float(opd_summary["wall_time_seconds"]),
            "gpu_hours": float(opd_summary["wall_time_seconds"]) * 4 / 3600,
            "peak_gpu_memory_bytes": int(opd_summary["peak_gpu_memory_bytes"]),
        },
    }
    rows = []
    for item in items:
        condition = item["condition"]
        summary = summaries[condition]
        lower, upper = wilson_interval(int(summary["correct"]), int(summary["count"]))
        rows.append(
            {
                "condition": condition,
                "label": LABELS[condition],
                "correct": int(summary["correct"]),
                "total": int(summary["count"]),
                "exact_accuracy": float(summary["exact_answer_accuracy"]),
                "wilson_lower": lower,
                "wilson_upper": upper,
                "selected_checkpoint": item["model"],
                "selected_step": int(item["checkpoint_step"]),
                "evaluation_generated_tokens": int(summary["generated_tokens"]),
                "evaluation_verifier_calls": int(summary["verifier_calls"]),
                "evaluation_wall_time_seconds": float(summary["wall_time_seconds"]),
                "evaluation_peak_gpu_memory_bytes": int(summary["peak_gpu_memory_bytes"]),
                **training[condition],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: list[dict[str, Any]]) -> None:
    teacher_gate = load_summary(ARTIFACT_ROOT / "development" / "teacher_gate.json")
    teacher_curve = teacher_gate["teacher_curve"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(
        [int(row["checkpoint_step"]) for row in teacher_curve],
        [float(row["exact_answer_accuracy"]) for row in teacher_curve],
        marker="o",
    )
    axes[0].set_title("Teacher development route")
    axes[0].set_xlabel("Optimizer updates")
    axes[0].set_ylabel("Exact-answer accuracy")
    axes[0].set_ylim(0, 1)
    x = np.arange(len(rows))
    accuracy = np.asarray([row["exact_accuracy"] for row in rows])
    lower = accuracy - np.asarray([row["wilson_lower"] for row in rows])
    upper = np.asarray([row["wilson_upper"] for row in rows]) - accuracy
    axes[1].bar(x, accuracy, color="#4C78A8")
    axes[1].errorbar(x, accuracy, yerr=[lower, upper], fmt="none", ecolor="black", capsize=3)
    axes[1].set_xticks(x, [row["label"] for row in rows], rotation=35, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Frozen final (79 problems)")
    axes[1].set_ylabel("Exact-answer accuracy")
    figure.tight_layout()
    figure.savefig(ARTIFACT_ROOT / "figure_gap.png", dpi=180)
    figure.savefig(ARTIFACT_ROOT / "figure_gap.pdf")
    plt.close(figure)


def _format_memory(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / (1024 ** 3):.1f} GiB"


def _format_seconds(value: float) -> str:
    hours, remainder = divmod(int(round(value)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def write_markdown_report(payload: dict[str, Any]) -> None:
    rows = payload["conditions"]
    teacher_gate = load_summary(ARTIFACT_ROOT / "development" / "teacher_gate.json")
    selection_path = TEACHER_ROUTE.parent / "learning_rate_selection.json"
    learning_rate_selection = load_summary(selection_path) if selection_path.exists() else {}
    scheduler_path = ARTIFACT_ROOT / "scheduler_state.json"
    scheduler = load_summary(scheduler_path) if scheduler_path.exists() else {}
    reported_scheduler_state = "COMPLETED" if payload.get("conditions") else scheduler.get("state", "RUNNING")
    run_root = Path("runs") / RUN_ID
    direct = load_summary(run_root / "student_direct_rlvr" / "checkpoint_selection.json")
    oracle = load_summary(run_root / "student_oracle" / "checkpoint_selection.json")
    opd = load_summary(run_root / "student_final_opd" / "checkpoint_selection.json")
    curve = teacher_gate["teacher_curve"]
    point = payload["point_statistics"]
    lower = payload["bootstrap"]["lower_bounds"]
    lines = [
        "# Minimal Acquisition-Gap Result",
        "",
        f"Run: `{payload['run_id']}`  ",
        f"Setting: `{payload['benchmark']} × {TEACHER_ID} / {STUDENT_ID}`  ",
        f"Frozen final split: {payload['frozen_final_problems']} problems; official metric: exact-answer accuracy.",
        "",
        "## Frozen final results",
        "",
        "| Condition | Correct | Accuracy (95% Wilson CI) | Updates | Checkpoint | Generated tokens | Teacher-scored tokens | Verifier calls | Wall time | GPU-hours | Peak GPU memory |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {correct}/{total} | {accuracy:.3f} [{lower:.3f}, {upper:.3f}] | "
            "{updates} | `{checkpoint}` | {generated:,} | {teacher_tokens:,} | {verifier:,} | "
            "{wall} | {gpu_hours:.2f} | {memory} |".format(
                label=row["label"], correct=row["correct"], total=row["total"],
                accuracy=row["exact_accuracy"], lower=row["wilson_lower"], upper=row["wilson_upper"],
                updates=row["updates"], checkpoint=row["selected_checkpoint"],
                generated=row["generated_tokens"], teacher_tokens=row["teacher_scored_tokens"],
                verifier=row["verifier_calls"], wall=_format_seconds(float(row["wall_time_seconds"])),
                gpu_hours=float(row["gpu_hours"]), memory=_format_memory(row["peak_gpu_memory_bytes"]),
            )
        )
    lines.extend(
        [
            "",
            "The training-token columns exclude frozen-final evaluation tokens. Teacher peak memory is the highest audited per-rank `nvidia-smi` sample; minimal-pipeline stages use allocator peak memory.",
            "",
            "Frozen-final evaluation audit:",
            "",
            "| Condition | Evaluation tokens | Evaluation verifier calls | Evaluation wall time | Evaluation peak memory |",
            "|---|---:|---:|---:|---:|",
            *[
                f"| {row['label']} | {row['evaluation_generated_tokens']:,} | "
                f"{row['evaluation_verifier_calls']:,} | {_format_seconds(float(row['evaluation_wall_time_seconds']))} | "
                f"{_format_memory(row['evaluation_peak_gpu_memory_bytes'])} |"
                for row in rows
            ],
            "",
            "## Development selection",
            "",
            "Teacher curve (updates: correct/21): "
            + ", ".join(f"{int(row['checkpoint_step'])}: {int(row['correct'])}/21" for row in curve)
            + ".",
            "",
            "Student Direct RLVR curve (updates: correct/21): "
            + ", ".join(
                f"{step}: {int(summary['correct'])}/21"
                for step, summary in sorted(direct["candidate_summaries"].items(), key=lambda pair: int(pair[0]))
            )
            + ".",
            "",
            "Student Oracle curve (updates: correct/21): 0: "
            f"{int(teacher_gate['student_s0']['correct'])}/21, "
            + ", ".join(
                f"{step}: {int(summary['correct'])}/21"
                for step, summary in sorted(oracle["candidate_summaries"].items(), key=lambda pair: int(pair[0]))
            )
            + ".",
            "",
            "Student Final OPD curve (updates: correct/21): "
            + ", ".join(
                f"{step}: {int(summary['correct'])}/21"
                for step, summary in sorted(opd["candidate_summaries"].items(), key=lambda pair: int(pair[0]))
            )
            + ".",
            "",
            f"Teacher learning rate: `{learning_rate_selection.get('selected_learning_rate', 'unavailable')}`; "
            f"probe scores: `{json.dumps(learning_rate_selection.get('candidate_scores', {}), sort_keys=True)}`.",
            "",
            "Student learning rates: Direct RLVR `1e-6`; Oracle Supervision `2e-6`; Final OPD `2e-6`.",
            "",
            f"Teacher revision: `{TEACHER_REVISION}`; Student revision: `{STUDENT_REVISION}`; "
            "OlymMATH revision: `94e19ffb8b7b7abe0f10f0015620c2e1eb57bf67`.",
            "",
            f"Training-data revision: `{payload['revisions']['math_training_dataset']}`; "
            f"frozen development SHA-256: `{payload['revisions']['development_sha256']}`; "
            f"frozen final SHA-256: `{payload['revisions']['final_sha256']}`; "
            f"Student stream SHA-256: `{payload['revisions']['student_stream_sha256']}`.",
            "",
            "## Paired inference",
            "",
            f"Point estimates: Delta_T={point['Delta_T']:.4f}, Delta_O={point['Delta_O']:.4f}, G={point['G']:.4f}.",
            "",
            "Shared-index paired bootstrap: 10,000 replicates, seed 42. Bonferroni one-sided 95% simultaneous lower bounds "
            f"(1.67th percentile): Delta_T={lower['Delta_T']:.4f}, Delta_O={lower['Delta_O']:.4f}, G={lower['G']:.4f}.",
            "",
            f"Decision: **{payload['decision']}**",
            "",
            payload["statement"],
            "",
            "## Recovery record",
            "",
            f"Scheduler state: `{reported_scheduler_state}`. "
            f"Stage attempts: `{json.dumps(scheduler.get('stages', {}), sort_keys=True)}`.",
            "",
            "Artifacts: `final_result.json`, `final_scores.csv`, `figure_gap.png`, `figure_gap.pdf`, and per-condition raw predictions/verifier records under `final/`.",
            "",
        ]
    )
    (ARTIFACT_ROOT / "gap_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    items = build_final_items()
    evaluate_items(items, FINAL_TEST, ARTIFACT_ROOT / "plans" / "frozen_final.json")
    summaries = {item["condition"]: load_summary(Path(item["summary"])) for item in items}
    prompt_ids, correctness = load_aligned_correctness(items)
    point = {
        "Delta_T": float(correctness["teacher_t400"].mean() - correctness["teacher_t0"].mean()),
        "Delta_O": float(correctness["student_oracle"].mean() - correctness["student_s0"].mean()),
        "G": float(
            min(correctness["teacher_t400"].mean(), correctness["student_oracle"].mean())
            - max(correctness["student_direct_rlvr"].mean(), correctness["student_final_opd"].mean())
        ),
    }
    bootstrap = paired_bootstrap(correctness)
    lower = bootstrap["lower_bounds"]
    if all(float(lower[name]) > 0 for name in ("Delta_T", "Delta_O", "G")):
        decision = "EMPIRICAL_ACQUISITION_GAP_EVIDENCE"
        statement = (
            "Under the frozen OlymMATH English Easy setting, model pair, training budgets, "
            "and this single training realization, the results provide empirical evidence of an acquisition gap."
        )
    elif all(point[name] > 0 for name in ("Delta_T", "Delta_O", "G")):
        decision = "CANDIDATE_PATTERN_NOT_CONCLUSIVE"
        statement = (
            "The candidate acquisition-gap pattern was observed, but the 79-item final split "
            "does not provide conclusive statistical evidence."
        )
    else:
        decision = "ACQUISITION_GAP_NOT_ESTABLISHED"
        statement = "An acquisition gap was not established in this setting."
    rows = cost_rows(items, summaries)
    write_csv(ARTIFACT_ROOT / "final_scores.csv", rows)
    payload = {
        "run_id": RUN_ID,
        "benchmark": "OlymMATH English Easy",
        "development_problems": 21,
        "frozen_final_problems": len(prompt_ids),
        "primary_metric": "exact_answer_accuracy",
        "training_seed": 42,
        "revisions": {
            "teacher_model": TEACHER_REVISION,
            "student_model": STUDENT_REVISION,
            "olymmath": "94e19ffb8b7b7abe0f10f0015620c2e1eb57bf67",
            "math_training_dataset": "31dd309567e3da778038cc87d868b6097a3ccf68",
            "development_sha256": sha256_file(ROOT / "data" / "frozen" / "olymmath" / "development.jsonl"),
            "final_sha256": sha256_file(FINAL_TEST),
            "student_stream_sha256": sha256_file(STUDENT_STREAM),
            "teacher_stream_sha256": sha256_file(
                ROOT / "data" / "frozen" / "math_training" / "teacher_prompt_stream.jsonl"
            ),
        },
        "point_statistics": point,
        "bootstrap": bootstrap,
        "decision": decision,
        "statement": statement,
        "conditions": rows,
    }
    atomic_json(ARTIFACT_ROOT / "final_result.json", payload)
    make_figure(rows)
    write_markdown_report(payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
