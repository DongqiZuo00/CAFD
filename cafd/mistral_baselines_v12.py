"""Matched nested-validation Direct RLVR and Endpoint OPD baselines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import mistral_progressive_observation_v11 as progressive
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

PROTOCOL = "mistral_acquisition_baseline_v12"
RUN_IDS = {
    "direct": "mistral_direct_observation_v12",
    "endpoint": "mistral_endpoint_observation_v12",
}
MILESTONES = [0, 10, 40, 80, 120, 160, 200]
TEACHER_COUNTS = [6, 4, 4, 2, 2]
_base_load_config = train_student.load_experiment_config
_active_config: dict[str, Any] | None = None


def _materialize_identity(config: dict[str, Any]) -> dict[str, Any]:
    method = os.environ.get("CAFD_BASELINE_METHOD", "")
    if method not in RUN_IDS:
        raise RuntimeError(f"invalid CAFD_BASELINE_METHOD: {method!r}")
    config["protocol"] = PROTOCOL
    config["experiment"]["run_id"] = RUN_IDS[method]
    config["baseline"] = {"method": method}
    if method == "direct":
        config["cafd"]["rollout_support"] = "student_on_policy"
        config["cafd"]["teacher_rollouts_per_prompt_by_phase"] = [0, 0, 0, 0, 0]
        config["cafd"]["target"] = "verifier_reward"
        config["cafd"]["objective"] = "hierarchical_rlvr"
    else:
        config["cafd"]["rollout_support"] = "matched_teacher_student"
        config["cafd"]["teacher_rollouts_per_prompt_by_phase"] = TEACHER_COUNTS
        config["cafd"]["target"] = "absolute_teacher_endpoint"
    config["final"]["conditions"] = [method]
    return config


def validate_config(config: dict[str, Any]) -> None:
    method = str(config["baseline"]["method"])
    identity = (str(config["protocol"]), str(config["experiment"]["run_id"]))
    if method not in RUN_IDS or identity != (PROTOCOL, RUN_IDS[method]):
        raise RuntimeError(f"wrong v12 baseline identity/method: {identity}, {method}")
    expected = {
        "teacher": (TEACHER_ID, TEACHER_REVISION),
        "student": (STUDENT_ID, STUDENT_REVISION),
    }
    for role, pinned in expected.items():
        record = config["models"][role]
        actual = (str(record["id"]), str(record["revision"]))
        if actual != pinned or "qwen" in actual[0].lower():
            raise RuntimeError(f"wrong pinned Mistral {role}: {actual}")
    if config["generation"]["prompt_renderer"] != PROMPT_RENDERER_MISTRAL3_INSTRUCT:
        raise RuntimeError("v12 requires the frozen Mistral renderer")
    if int(config["generation"]["max_new_tokens"]) != 2048:
        raise RuntimeError("v12 requires max_new_tokens=2048")
    settings = config["student"]
    if (
        int(settings["updates"]) != 200
        or int(settings["prompts_per_update"]) != 4
        or int(settings["rollouts_per_prompt"]) != 8
        or [int(step) for step in settings["milestones"]] != MILESTONES
    ):
        raise RuntimeError("v12 Student budget changed")
    generalization = config["generalization"]
    if (
        str(generalization["fit_split"]) != "data/cafd/generalization_v10/fit.jsonl"
        or str(generalization["selection_split"]) != "data/cafd/generalization_v10/selection.jsonl"
        or str(generalization["confirmation_split"]) != "data/cafd/development.jsonl"
        or int(generalization["fit_size"]) != 614
        or int(generalization["selection_size"]) != 64
        or int(generalization["confirmation_size"]) != 64
    ):
        raise RuntimeError("v12 nested-validation contract changed")
    if int(config["final"]["official_held_out_evaluations"]) != 0:
        raise RuntimeError("v12 must not access frozen test")
    support = str(config["cafd"]["rollout_support"])
    if method == "direct" and support != "student_on_policy":
        raise RuntimeError("Direct RLVR must remain on-policy")
    if method == "endpoint":
        counts = [int(x) for x in config["cafd"]["teacher_rollouts_per_prompt_by_phase"]]
        if support != "matched_teacher_student" or counts != TEACHER_COUNTS:
            raise RuntimeError("Endpoint OPD support schedule changed")


def _load_config(root: Path) -> dict[str, Any]:
    global _active_config
    config = _materialize_identity(_base_load_config(root))
    validate_config(config)
    _active_config = config
    return config


def _load_rows(root: Path, split: str):
    if _active_config is None:
        raise RuntimeError("v12 config was not loaded before rows")
    if split == "train":
        return load_nested_rows(root, "fit")
    if split == "development":
        return load_nested_rows(root, "selection")
    raise RuntimeError(f"v12 refuses undeclared split: {split}")


def _teacher_set(method: str, layout, device):
    expected = str(_active_config["baseline"]["method"]) if _active_config else ""
    if method != expected:
        raise RuntimeError(f"v12 method mismatch: {method} != {expected}")
    if method == "direct":
        return {}
    return progressive._teacher_route("progressive", layout, device)


def _rollouts(*args, **kwargs):
    if _active_config and _active_config["baseline"]["method"] == "endpoint":
        return progressive._support_rollouts(*args, **kwargs)
    return progressive._base_generate_rollouts(*args, **kwargs)


def install_baseline() -> None:
    install_into(train_student)
    train_student.load_experiment_config = _load_config
    train_student.load_rows = _load_rows
    train_student._teacher_set = _teacher_set
    train_student._generate_rollouts = _rollouts


if __name__ == "__main__":
    install_baseline()
    train_student.main()
