"""Nested-validation CAFD ablation for rollout-support generalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import train_student
from .generalization_split import load_nested_rows
from .mistral_cafd_support_full import NoHashPhaseController, _reference_identity
from .mistral_runtime import (
    PROMPT_RENDERER_MISTRAL3_INSTRUCT,
    STUDENT_ID,
    STUDENT_REVISION,
    TEACHER_ID,
    TEACHER_REVISION,
    install_into,
)


PROTOCOL = "mistral_cafd_generalization_v10"
RUN_IDS = {
    "next_teacher_replay": "mistral_cafd_gen_v10_teacher",
    "mixed_teacher_student": "mistral_cafd_gen_v10_mixed",
}
PHASE_UPDATES = 40
TOTAL_UPDATES = 200
MILESTONES = [0, 10, 40, 80, 120, 160, 200]
_base_generate_rollouts = train_student._generate_rollouts
_base_load_experiment_config = train_student.load_experiment_config
_support_models: dict[str, Any] = {}
_active_config: dict[str, Any] | None = None


def _teacher_counts(config: dict[str, Any]) -> list[int]:
    counts = [int(value) for value in config["cafd"]["teacher_rollouts_per_prompt_by_phase"]]
    if len(counts) != 5 or any(value < 0 or value > 8 for value in counts):
        raise RuntimeError(f"invalid mixed-support schedule: {counts}")
    return counts


def validate_config(config: dict[str, Any]) -> None:
    support = str(config["cafd"]["rollout_support"])
    identity = (str(config["protocol"]), str(config["experiment"]["run_id"]))
    if support not in RUN_IDS or identity != (PROTOCOL, RUN_IDS[support]):
        raise RuntimeError(f"wrong v10 identity/support: {identity}, {support}")
    expected_models = {
        "teacher": (TEACHER_ID, TEACHER_REVISION),
        "student": (STUDENT_ID, STUDENT_REVISION),
    }
    for role, expected in expected_models.items():
        record = config["models"][role]
        actual = (str(record["id"]), str(record["revision"]))
        if actual != expected or "qwen" in actual[0].lower():
            raise RuntimeError(f"wrong pinned Mistral {role}: {actual}")
    if config["generation"]["prompt_renderer"] != PROMPT_RENDERER_MISTRAL3_INSTRUCT:
        raise RuntimeError("v10 requires the frozen Mistral renderer")
    if int(config["generation"]["max_new_tokens"]) != 2048:
        raise RuntimeError("v10 requires max_new_tokens=2048")
    settings = config["student"]
    if int(settings["updates"]) != TOTAL_UPDATES:
        raise RuntimeError("v10 requires exactly 200 updates")
    if int(settings["prompts_per_update"]) != 4 or int(settings["rollouts_per_prompt"]) != 8:
        raise RuntimeError("v10 requires 4 prompts x 8 rollouts")
    if [int(step) for step in settings["milestones"]] != MILESTONES:
        raise RuntimeError("v10 development milestones changed")
    if int(config["cafd"]["phase_updates"]) != PHASE_UPDATES:
        raise RuntimeError("v10 requires five 40-update phases")
    counts = _teacher_counts(config)
    expected = [8, 8, 8, 8, 8] if support == "next_teacher_replay" else [6, 4, 4, 2, 2]
    if counts != expected:
        raise RuntimeError(f"unexpected v10 support schedule: {counts}")
    generalization = config["generalization"]
    if (
        str(generalization["fit_split"]) != "data/cafd/generalization_v10/fit.jsonl"
        or str(generalization["selection_split"]) != "data/cafd/generalization_v10/selection.jsonl"
        or str(generalization["confirmation_split"]) != "data/cafd/development.jsonl"
        or int(generalization["fit_size"]) != 614
        or int(generalization["selection_size"]) != 64
        or int(generalization["confirmation_size"]) != 64
    ):
        raise RuntimeError("v10 nested-validation contract changed")
    if int(config["final"]["official_held_out_evaluations"]) != 0:
        raise RuntimeError("v10 diagnostic must not access frozen test")


def _load_config(root: Path) -> dict[str, Any]:
    global _active_config
    config = _base_load_experiment_config(root)
    validate_config(config)
    _active_config = config
    return config


def _load_rows(root: Path, split: str):
    if _active_config is None:
        raise RuntimeError("v10 config was not loaded before rows")
    if split == "train":
        return load_nested_rows(root, "fit")
    if split == "development":
        return load_nested_rows(root, "selection")
    raise RuntimeError(f"v10 refuses undeclared split: {split}")


def _teacher_route(method: str, layout, device):
    global _support_models
    if method != "cafd":
        raise RuntimeError("v10 executes CAFD only")
    route = json.loads((layout.teacher / "route.json").read_text(encoding="utf-8"))
    checkpoints = [item["checkpoint"] for item in route["checkpoints"]]
    if (
        route.get("status") != "frozen"
        or route.get("kind") != "mistral_full_acquisition_raw_instruct_to_sft_to_rl"
        or len(checkpoints) != 6
    ):
        raise RuntimeError("v10 requires the frozen six-checkpoint Mistral Teacher route")
    _support_models = {
        f"R{index}": train_student._load_teacher(layout.root, checkpoint, device)
        for index, checkpoint in enumerate(checkpoints)
    }
    return _support_models


def _tag(items, source: str, phase_index: int):
    for item in items:
        item["rollout_source"] = source
        item["support_phase"] = phase_index
    return items


def _support_rollouts(
    model,
    tokenizer,
    rows,
    indices,
    update,
    settings,
    max_new_tokens,
    context,
    prompt_renderer,
):
    if _active_config is None:
        raise RuntimeError("v10 support schedule is unavailable")
    phase_index = int(update) // PHASE_UPDATES
    source_key = f"R{phase_index + 1}"
    teacher_count = _teacher_counts(_active_config)[phase_index]
    student_count = int(settings["rollouts_per_prompt"]) - teacher_count
    result = []
    if teacher_count:
        teacher_settings = dict(settings)
        teacher_settings["rollouts_per_prompt"] = teacher_count
        result.extend(_tag(
            _base_generate_rollouts(
                _support_models[source_key], tokenizer, rows, indices, update,
                teacher_settings, max_new_tokens, context, prompt_renderer,
            ),
            f"{source_key}_next_teacher", phase_index,
        ))
    if student_count:
        student_settings = dict(settings)
        student_settings["rollouts_per_prompt"] = student_count
        result.extend(_tag(
            _base_generate_rollouts(
                model, tokenizer, rows, indices, update, student_settings,
                max_new_tokens, context, prompt_renderer,
            ),
            "student_on_policy", phase_index,
        ))
    expected = len(indices) * int(settings["rollouts_per_prompt"])
    if len(result) != expected:
        raise RuntimeError(f"support mixer produced {len(result)} slots, expected {expected}")
    by_prompt: dict[int, int] = {}
    for item in result:
        slot = int(item["prompt_slot"])
        item["sample_slot"] = by_prompt.get(slot, 0)
        by_prompt[slot] = int(item["sample_slot"]) + 1
    if set(by_prompt.values()) != {int(settings["rollouts_per_prompt"])}:
        raise RuntimeError(f"support mixer broke prompt slot counts: {by_prompt}")
    return result


def install_v10() -> None:
    install_into(train_student)
    train_student.load_experiment_config = _load_config
    train_student.load_rows = _load_rows
    train_student._teacher_set = _teacher_route
    train_student._generate_rollouts = _support_rollouts
    train_student.PhaseController = NoHashPhaseController
    train_student.parameter_sha256 = _reference_identity


if __name__ == "__main__":
    install_v10()
    train_student.main()
