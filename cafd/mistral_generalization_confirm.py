"""One-shot confirmation on the original development split; never reads frozen test."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = load_experiment_config(root)
    if int(config["final"]["official_held_out_evaluations"]) != 0:
        raise RuntimeError("confirmation entrypoint refuses configs that permit frozen test")
    if str(config["generalization"]["confirmation_split"]) != "data/cafd/development.jsonl":
        raise RuntimeError("confirmation split changed")
    layout = experiment_layout(root, config)
    output = layout.method("cafd") / "confirmation.json"
    if output.exists():
        return
    selected = _load(layout.method("cafd") / "selected.json")
    checkpoint = Path(str(selected["checkpoint"])).resolve()
    if layout.method("cafd").resolve() not in checkpoint.parents:
        raise RuntimeError(f"selected checkpoint escaped run: {checkpoint}")

    context = init_distributed()
    if context.world_size != 1:
        raise RuntimeError("confirmation requires exactly one GPU")
    cache = root / ".cache" / "huggingface" / "hub"
    teacher_tokenizer, student_tokenizer = load_tokenizers(cache)
    student = load_model(
        str(checkpoint), "", cache_dir=cache, device=context.device, trainable=False
    )
    assert_exact_tokenizer_pair(
        teacher_tokenizer, student_tokenizer, student_model=student
    )
    model = HiddenCausalLM(student).eval().requires_grad_(False)
    rows = load_rows(root, "development")
    if len(rows) != int(config["generalization"]["confirmation_size"]):
        raise RuntimeError(f"unexpected confirmation size: {len(rows)}")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["problem_family"])].append(row)

    aggregate = {name: 0 for name in DETAIL_FIELDS}
    families: dict[str, dict[str, int | float]] = {}
    for family in sorted(groups):
        score = evaluate(
            model,
            student_tokenizer,
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
    if aggregate["total"] != 64 or aggregate["correct"] != aggregate["full_pass"]:
        raise RuntimeError(f"invalid confirmation aggregate: {aggregate}")
    selection_correct = round(float(selected["development_accuracy"]) * 64)
    payload = {
        "status": "confirmed",
        "protocol": str(config["protocol"]),
        "run_id": str(config["experiment"]["run_id"]),
        "rollout_support": str(config["cafd"]["rollout_support"]),
        "selection_split": str(config["generalization"]["selection_split"]),
        "confirmation_split": str(config["generalization"]["confirmation_split"]),
        "frozen_test_accessed": False,
        "selected_step": int(selected["selected_step"]),
        "checkpoint": str(checkpoint),
        "selection_correct": selection_correct,
        "selection_total": 64,
        "selection_accuracy": float(selected["development_accuracy"]),
        "confirmation": aggregate,
        "by_family": families,
        "selection_minus_confirmation": selection_correct - aggregate["correct"],
    }
    write_json(output, payload)
    write_json(layout.state_root / "confirmation.json", {
        "stage": "development_confirmation_complete",
        "correct": aggregate["correct"],
        "total": aggregate["total"],
        "frozen_test_accessed": False,
    })


if __name__ == "__main__":
    install_into(sys.modules[__name__])
    main()
