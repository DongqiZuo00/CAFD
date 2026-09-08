"""Development-only milestone observation for CAFD and matched Progressive GKD."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .configuration import load_experiment_config
from .data import load_rows
from .layout import experiment_layout
from .modeling import assert_exact_tokenizer_pair, load_model, load_tokenizers
from .mistral_runtime import install_into
from .training_common import HiddenCausalLM, evaluate, init_distributed, write_json


DETAIL_FIELDS = (
    "correct", "total", "generated_tokens", "eos_terminated", "token_limit_hits",
    "invalid_format", "format_only", "parse_only", "partial_pass", "full_pass",
    "contract_valid", "parse_or_better",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _auc(records: list[dict[str, Any]], key: str = "accuracy") -> float:
    ordered = sorted(records, key=lambda row: int(row["step"]))
    total_span = int(ordered[-1]["step"]) - int(ordered[0]["step"])
    if total_span <= 0:
        raise RuntimeError("observation curve has no positive span")
    area = 0.0
    for left, right in zip(ordered, ordered[1:]):
        width = int(right["step"]) - int(left["step"])
        area += width * (float(left[key]) + float(right[key])) / 2.0
    return area / total_span


def _evaluate_checkpoint(model, tokenizer, groups, config, context):
    aggregate = {name: 0 for name in DETAIL_FIELDS}
    families: dict[str, dict[str, int | float]] = {}
    for family in sorted(groups):
        score = evaluate(
            model,
            tokenizer,
            groups[family],
            max_prompt_length=2048,
            max_completion_length=int(config["generation"]["max_new_tokens"]),
            context=context,
            prompt_renderer=str(config["generation"]["prompt_renderer"]),
            include_details=True,
        )
        families[family] = score
        for name in DETAIL_FIELDS:
            aggregate[name] += int(score[name])
    aggregate["accuracy"] = aggregate["correct"] / aggregate["total"]
    return aggregate, families


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", choices=["cafd", "progressive"], required=True)
    parser.add_argument("--label", choices=["CAFD", "Progressive GKD"], required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = load_experiment_config(root)
    if int(config["final"]["official_held_out_evaluations"]) != 0:
        raise RuntimeError("observation refuses configs that permit frozen test")
    if str(config["generalization"]["confirmation_split"]) != "data/cafd/development.jsonl":
        raise RuntimeError("observation confirmation split changed")
    layout = experiment_layout(root, config)
    selected = _load(layout.method(args.method) / "selected.json")
    milestones = [int(step) for step in config["student"]["milestones"]]
    output = layout.artifact_root / "acquisition_observation.json"
    payload = _load(output) if output.exists() else {
        "status": "running",
        "label": args.label,
        "run_id": str(config["experiment"]["run_id"]),
        "method": args.method,
        "split": "data/cafd/development.jsonl",
        "frozen_test_accessed": False,
        "records": [],
    }
    if (
        payload["run_id"] != str(config["experiment"]["run_id"])
        or payload["method"] != args.method
        or payload.get("frozen_test_accessed") is not False
    ):
        raise RuntimeError("observation resume identity changed")
    completed = {int(record["step"]) for record in payload["records"]}

    context = init_distributed()
    if context.world_size != 1:
        raise RuntimeError("milestone observation requires exactly one GPU")
    cache = root / ".cache" / "huggingface" / "hub"
    teacher_tokenizer, student_tokenizer = load_tokenizers(cache)
    rows = load_rows(root, "development")
    if len(rows) != 64:
        raise RuntimeError(f"unexpected development size: {len(rows)}")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["problem_family"])].append(row)

    for step in milestones:
        if step in completed:
            continue
        checkpoint = (
            layout.student_base if step == 0
            else layout.method(args.method) / f"step{step}"
        )
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)
        student = load_model(
            str(checkpoint), "", cache_dir=cache,
            device=context.device, trainable=False,
        )
        assert_exact_tokenizer_pair(
            teacher_tokenizer, student_tokenizer, student_model=student
        )
        model = HiddenCausalLM(student).eval().requires_grad_(False)
        aggregate, families = _evaluate_checkpoint(
            model, student_tokenizer, groups, config, context
        )
        payload["records"].append({
            "step": step,
            **aggregate,
            "by_family": families,
        })
        payload["records"].sort(key=lambda row: int(row["step"]))
        write_json(output, payload)
        del model, student
        gc.collect()
        torch.cuda.empty_cache()

    records = sorted(payload["records"], key=lambda row: int(row["step"]))
    peak = max(records, key=lambda row: (int(row["correct"]), -int(row["step"])))
    final = records[-1]
    selected_step = int(selected["selected_step"])
    selected_record = next(row for row in records if int(row["step"]) == selected_step)
    payload.update({
        "status": "complete",
        "selected_step": selected_step,
        "selected_checkpoint": str(selected["checkpoint"]),
        "selected_record": selected_record,
        "peak_step": int(peak["step"]),
        "peak_correct": int(peak["correct"]),
        "final_correct": int(final["correct"]),
        "retention_from_peak_correct": int(final["correct"]) - int(peak["correct"]),
        "normalized_accuracy_auc": _auc(records),
        "frozen_test_accessed": False,
    })
    write_json(output, payload)

    csv_path = layout.artifact_root / "acquisition_observation.csv"
    fields = [
        "label", "step", "correct", "total", "accuracy", "contract_valid",
        "partial_pass", "full_pass", "invalid_format", "token_limit_hits",
        "contains_count_correct", "contains_ordered_correct",
        "contains_substring_correct",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "label": args.label,
                **{name: record[name] for name in (
                    "step", "correct", "total", "accuracy", "contract_valid",
                    "partial_pass", "full_pass", "invalid_format", "token_limit_hits",
                )},
                **{
                    f"{family}_correct": record["by_family"][family]["correct"]
                    for family in (
                        "contains_count", "contains_ordered", "contains_substring"
                    )
                },
            })


if __name__ == "__main__":
    install_into(sys.modules[__name__])
    main()
