"""Assemble the only formal CAFD-v1 scores, costs, plot, and report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from .configuration import load_experiment_config
from .layout import experiment_layout


ORDER = ["teacher_base", "teacher_rlvr", "student_base", "capacity", "direct", "endpoint", "progressive", "cafd"]


def wilson(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = correct / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = load_experiment_config(root)
    layout = experiment_layout(root, config)
    final_dir = layout.final
    records = [json.loads((final_dir / f"{key}.json").read_text(encoding="utf-8")) for key in ORDER]
    scores = []
    costs = []
    for record in records:
        low, high = wilson(int(record["correct"]), int(record["total"]))
        scores.append(
            {
                "condition": record["label"],
                "correct": int(record["correct"]),
                "total": int(record["total"]),
                "test_full_pass": float(record["accuracy"]),
                "ci95_low": low,
                "ci95_high": high,
                "selected_step": int(record["selected_step"]),
                "budget_updates": int(record["budget"]),
                "checkpoint": record["checkpoint"],
            }
        )
        logical_generated = int(
            record.get("logical_generated_tokens", record.get("generated_tokens", 0))
        )
        logical_teacher_scored = int(
            record.get(
                "logical_teacher_scored_tokens",
                record.get("teacher_scored_tokens", 0),
            )
        )
        costs.append(
            {
                "condition": record["label"],
                "generated_tokens": logical_generated,
                "teacher_scored_tokens": logical_teacher_scored,
                "logical_generated_tokens": logical_generated,
                "logical_teacher_scored_tokens": logical_teacher_scored,
                "actual_generated_tokens": int(
                    record.get("actual_generated_tokens", logical_generated)
                ),
                "actual_teacher_scored_tokens": int(
                    record.get("actual_teacher_scored_tokens", logical_teacher_scored)
                ),
                "gpu_hours": float(record.get("gpu_hours", 0.0)),
            }
        )
    artifact = layout.artifact_root
    _write_csv(
        artifact / "final_scores.csv",
        scores,
        ["condition", "correct", "total", "test_full_pass", "ci95_low", "ci95_high", "selected_step", "budget_updates", "checkpoint"],
    )
    _write_csv(
        artifact / "costs.csv",
        costs,
        [
            "condition",
            "generated_tokens",
            "teacher_scored_tokens",
            "logical_generated_tokens",
            "logical_teacher_scored_tokens",
            "actual_generated_tokens",
            "actual_teacher_scored_tokens",
            "gpu_hours",
        ],
    )

    labels = [row["condition"] for row in scores]
    values = [row["test_full_pass"] for row in scores]
    lows = [value - row["ci95_low"] for value, row in zip(values, scores, strict=True)]
    highs = [row["ci95_high"] - value for value, row in zip(values, scores, strict=True)]
    fig, axis = plt.subplots(figsize=(11, 5.8))
    colors = ["#5B6C8F", "#3D5A80", "#8D99AE", "#6A994E", "#BC6C25", "#9B5DE5", "#2A9D8F", "#D62828"]
    axis.bar(range(len(labels)), values, color=colors, yerr=[lows, highs], capsize=4)
    threshold = 52 / 64
    axis.axhline(threshold, color="#222222", linestyle="--", linewidth=1.2, label="capacity threshold 52/64")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Official held-out full-pass / pass@1")
    axis.set_xticks(range(len(labels)), labels, rotation=24, ha="right")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(artifact / "result.png", dpi=220)
    plt.close(fig)

    joined = {row["condition"]: row for row in scores}
    gap = (
        joined["Teacher Base"]["test_full_pass"] < threshold <= joined["Teacher RLVR"]["test_full_pass"]
        and joined["Student Base"]["test_full_pass"] < threshold <= joined["Capacity Control"]["test_full_pass"]
        and joined["Direct RLVR"]["test_full_pass"] < threshold
        and joined["Endpoint OPD"]["test_full_pass"] < threshold
    )
    cafd_closes = joined["CAFD-v1"]["test_full_pass"] >= threshold
    progressive_closes = joined["Progressive GKD"]["test_full_pass"] >= threshold
    lines = [
        "# CAFD-v1 — Manufactoria-HAS",
        "",
        "Single-seed (2027), single benchmark/backbone setting. Wilson intervals describe official-test task uncertainty only, not training-seed variance.",
        "",
        "| Condition | Correct/N | Test full-pass | 95% CI | Selected step | Generated tokens (logical/actual) | Teacher-scored tokens (logical/actual) | GPU-hours |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cost_map = {row["condition"]: row for row in costs}
    for score in scores:
        cost = cost_map[score["condition"]]
        lines.append(
            f"| {score['condition']} | {score['correct']}/{score['total']} | {score['test_full_pass']:.4f} | "
            f"[{score['ci95_low']:.4f}, {score['ci95_high']:.4f}] | {score['selected_step']} | "
            f"{cost['logical_generated_tokens']}/{cost['actual_generated_tokens']} | "
            f"{cost['logical_teacher_scored_tokens']}/{cost['actual_teacher_scored_tokens']} | {cost['gpu_hours']:.3f} |"
        )
    lines.extend(["", f"Preregistered acquisition gap: **{'supported' if gap else 'not supported'}**."])
    lines.append(f"CAFD-v1 threshold closure: **{'yes' if cafd_closes else 'no'}**.")
    if progressive_closes:
        lines.append("Progressive GKD also reaches the threshold; this setting therefore does not establish superiority over trajectory distillation.")
    (artifact / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
