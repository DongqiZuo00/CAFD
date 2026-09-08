"""Matched-support Progressive GKD control for the CAFD mechanism observation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import train_student
from .generalization_split import load_nested_rows
from .mistral_runtime import (
    PROMPT_RENDERER_MISTRAL3_INSTRUCT,
    STUDENT_ID,
    STUDENT_REVISION,
    TEACHER_ID,
    TEACHER_REVISION,
    install_into,
)


PROTOCOL = "mistral_acquisition_observation_v11"
RUN_ID = "mistral_progressive_gkd_observation_v11"
PHASE_UPDATES = 40
TOTAL_UPDATES = 200
MILESTONES = [0, 10, 40, 80, 120, 160, 200]
TEACHER_COUNTS = [6, 4, 4, 2, 2]
_base_generate_rollouts = train_student._generate_rollouts
_base_load_experiment_config = train_student.load_experiment_config
_base_teacher_set = train_student._teacher_set
_active_config: dict[str, Any] | None = None
_support_models: dict[str, Any] = {}


def validate_config(config: dict[str, Any]) -> None:
    identity = (str(config["protocol"]), str(config["experiment"]["run_id"]))
    if identity != (PROTOCOL, RUN_ID):
        raise RuntimeError(f"wrong observation identity: {identity}")
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
        raise RuntimeError("observation requires the frozen Mistral renderer")
    if int(config["generation"]["max_new_tokens"]) != 2048:
        raise RuntimeError("observation requires max_new_tokens=2048")
    settings = config["student"]
    if (
        int(settings["updates"]) != TOTAL_UPDATES
        or int(settings["prompts_per_update"]) != 4
        or int(settings["rollouts_per_prompt"]) != 8
        or [int(step) for step in settings["milestones"]] != MILESTONES
    ):
        raise RuntimeError("observation training budget changed")
    if int(config["cafd"]["phase_updates"]) != PHASE_UPDATES:
        raise RuntimeError("observation requires five 40-update phases")
    if [int(value) for value in config["cafd"]["teacher_rollouts_per_prompt_by_phase"]] != TEACHER_COUNTS:
        raise RuntimeError("observation support schedule changed")
    if str(config["cafd"]["rollout_support"]) != "matched_teacher_student":
        raise RuntimeError("observation requires matched Teacher/Student support")
    if str(config["cafd"]["target"]) != "absolute_next_teacher_checkpoint":
        raise RuntimeError("Progressive GKD target changed")
    generalization = config["generalization"]
    if (
        str(generalization["fit_split"]) != "data/cafd/generalization_v10/fit.jsonl"
        or str(generalization["selection_split"]) != "data/cafd/generalization_v10/selection.jsonl"
        or str(generalization["confirmation_split"]) != "data/cafd/development.jsonl"
        or int(generalization["fit_size"]) != 614
        or int(generalization["selection_size"]) != 64
        or int(generalization["confirmation_size"]) != 64
    ):
        raise RuntimeError("observation nested-validation contract changed")
    if int(config["final"]["official_held_out_evaluations"]) != 0:
        raise RuntimeError("observation must not access frozen test")


def _load_config(root: Path) -> dict[str, Any]:
    global _active_config
    config = _base_load_experiment_config(root)
    validate_config(config)
    _active_config = config
    return config


def _load_rows(root: Path, split: str):
    if _active_config is None:
        raise RuntimeError("observation config was not loaded before rows")
    if split == "train":
        return load_nested_rows(root, "fit")
    if split == "development":
        return load_nested_rows(root, "selection")
    raise RuntimeError(f"observation refuses undeclared split: {split}")


def _teacher_route(method: str, layout, device):
    global _support_models
    if method != "progressive":
        raise RuntimeError("observation entrypoint executes Progressive GKD only")
    _support_models = _base_teacher_set(method, layout, device)
    if set(_support_models) != {"R1", "R2", "R3", "R4", "R5"}:
        raise RuntimeError(f"unexpected Progressive route: {sorted(_support_models)}")
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
    phase_index = int(update) // PHASE_UPDATES
    source_key = f"R{phase_index + 1}"
    teacher_count = TEACHER_COUNTS[phase_index]
    student_count = int(settings["rollouts_per_prompt"]) - teacher_count
    teacher_settings = dict(settings)
    teacher_settings["rollouts_per_prompt"] = teacher_count
    student_settings = dict(settings)
    student_settings["rollouts_per_prompt"] = student_count
    result = _tag(
        _base_generate_rollouts(
            _support_models[source_key], tokenizer, rows, indices, update,
            teacher_settings, max_new_tokens, context, prompt_renderer,
        ),
        f"{source_key}_next_teacher", phase_index,
    )
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


def install_observation() -> None:
    install_into(train_student)
    train_student.load_experiment_config = _load_config
    train_student.load_rows = _load_rows
    train_student._teacher_set = _teacher_route
    train_student._generate_rollouts = _support_rollouts


if __name__ == "__main__":
    install_observation()
    train_student.main()
