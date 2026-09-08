"""Isolated ten-update CAFD support probe using next-Teacher replay prefixes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import train_student
from .mistral_runtime import (
    PROMPT_RENDERER_MISTRAL3_INSTRUCT,
    STUDENT_ID,
    STUDENT_REVISION,
    TEACHER_ID,
    TEACHER_REVISION,
    install_into,
)


PROTOCOL = "mistral_cafd_support_probe_v8"
RUN_ID = "mistral_cafd_support_probe_v8"
_base_generate_rollouts = train_student._generate_rollouts
_base_load_experiment_config = train_student.load_experiment_config
_support_model = None


def validate_probe_config(config: dict[str, Any]) -> None:
    identity = (str(config["protocol"]), str(config["experiment"]["run_id"]))
    if identity != (PROTOCOL, RUN_ID):
        raise RuntimeError(f"wrong support-probe identity: {identity}")
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
        raise RuntimeError("support probe requires the frozen Mistral renderer")
    if int(config["student"]["updates"]) != 10:
        raise RuntimeError("support probe must stop after exactly 10 updates")
    if [int(step) for step in config["student"]["milestones"]] != [0, 10]:
        raise RuntimeError("support probe permits only baseline and step10 evaluations")
    if int(config["cafd"]["phase_updates"]) != 40:
        raise RuntimeError("support probe must remain inside phase 0")
    if config["cafd"].get("rollout_support") != "next_teacher_replay":
        raise RuntimeError("support probe requires next-Teacher replay")


def _load_probe_config(root: Path) -> dict[str, Any]:
    config = _base_load_experiment_config(root)
    validate_probe_config(config)
    return config


def _teacher_pair(method: str, layout, device):
    global _support_model
    if method != "cafd":
        raise RuntimeError("support probe runs CAFD only")
    route = json.loads((layout.teacher / "route.json").read_text(encoding="utf-8"))
    checkpoints = [item["checkpoint"] for item in route["checkpoints"]]
    if route.get("status") != "frozen" or len(checkpoints) != 6:
        raise RuntimeError("support probe requires the frozen six-checkpoint Teacher route")
    teachers = {
        "R0": train_student._load_teacher(layout.root, checkpoints[0], device),
        "R1": train_student._load_teacher(layout.root, checkpoints[1], device),
    }
    _support_model = teachers["R1"]
    return teachers


def _teacher_replay_rollouts(
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
    if _support_model is None:
        raise RuntimeError("next-Teacher replay model was not loaded")
    rollouts = _base_generate_rollouts(
        _support_model,
        tokenizer,
        rows,
        indices,
        update,
        settings,
        max_new_tokens,
        context,
        prompt_renderer,
    )
    for item in rollouts:
        item["rollout_source"] = "R1_next_teacher_replay"
    return rollouts


def install_support_probe() -> None:
    install_into(train_student)
    train_student.load_experiment_config = _load_probe_config
    train_student._teacher_set = _teacher_pair
    train_student._generate_rollouts = _teacher_replay_rollouts


if __name__ == "__main__":
    install_support_probe()
    train_student.main()
