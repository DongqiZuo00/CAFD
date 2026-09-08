"""Full support-corrected Mistral CAFD run with phase-stationary Teacher replay."""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from . import train_student
from .mistral_runtime import (
    PROMPT_RENDERER_MISTRAL3_INSTRUCT,
    STUDENT_ID,
    STUDENT_REVISION,
    TEACHER_ID,
    TEACHER_REVISION,
    install_into,
)
from .phase_controller import PhaseState, clone_frozen_model


PROTOCOL = "mistral_cafd_support_full_v9"
RUN_ID = "mistral_cafd_support_full_v9"
PHASE_UPDATES = 40
TOTAL_UPDATES = 200
MILESTONES = [0, 10, 40, 80, 120, 160, 200]
_base_generate_rollouts = train_student._generate_rollouts
_base_load_experiment_config = train_student.load_experiment_config
_base_phase_controller = train_student.PhaseController
_support_models: dict[str, Any] = {}


def validate_full_config(config: dict[str, Any]) -> None:
    identity = (str(config["protocol"]), str(config["experiment"]["run_id"]))
    if identity != (PROTOCOL, RUN_ID):
        raise RuntimeError(f"wrong full-run identity: {identity}")
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
        raise RuntimeError("full run requires the frozen Mistral renderer")
    if int(config["generation"]["max_new_tokens"]) != 2048:
        raise RuntimeError("full run requires max_new_tokens=2048")
    settings = config["student"]
    if int(settings["updates"]) != TOTAL_UPDATES:
        raise RuntimeError("full run requires exactly 200 updates")
    if int(settings["prompts_per_update"]) != 4 or int(settings["rollouts_per_prompt"]) != 8:
        raise RuntimeError("full run requires 4 prompts x 8 rollouts")
    if [int(step) for step in settings["milestones"]] != MILESTONES:
        raise RuntimeError("full run development milestones changed")
    if int(config["cafd"]["phase_updates"]) != PHASE_UPDATES:
        raise RuntimeError("full run requires five 40-update phases")
    if config["cafd"].get("rollout_support") != "next_teacher_replay":
        raise RuntimeError("full run requires next-Teacher replay")


def _load_full_config(root: Path) -> dict[str, Any]:
    config = _base_load_experiment_config(root)
    validate_full_config(config)
    return config


def _teacher_route(method: str, layout, device):
    global _support_models
    if method != "cafd":
        raise RuntimeError("support-corrected full run executes CAFD only")
    route = json.loads((layout.teacher / "route.json").read_text(encoding="utf-8"))
    checkpoints = [item["checkpoint"] for item in route["checkpoints"]]
    if (
        route.get("status") != "frozen"
        or route.get("kind") != "mistral_full_acquisition_raw_instruct_to_sft_to_rl"
        or len(checkpoints) != 6
    ):
        raise RuntimeError("full run requires the frozen six-checkpoint Mistral Teacher route")
    teachers = {
        f"R{index}": train_student._load_teacher(layout.root, checkpoint, device)
        for index, checkpoint in enumerate(checkpoints)
    }
    _support_models = teachers
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
    phase_index = int(update) // PHASE_UPDATES
    source_key = f"R{phase_index + 1}"
    if source_key not in _support_models:
        raise RuntimeError(f"next-Teacher replay model {source_key} was not loaded")
    rollouts = _base_generate_rollouts(
        _support_models[source_key],
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
        item["rollout_source"] = f"{source_key}_next_teacher_replay"
        item["support_phase"] = phase_index
    return rollouts


def _reference_identity(reference) -> str:
    identity = getattr(reference, "_cafd_reference_id", None)
    if not isinstance(identity, str):
        raise RuntimeError("phase reference lacks its immutable identity")
    if reference.training:
        raise RuntimeError("phase reference entered training mode")
    for parameter in reference.parameters():
        if parameter.requires_grad or parameter.grad is not None:
            raise RuntimeError("phase reference received gradients")
    return identity


class NoHashPhaseController:
    """Persist exact phase references while avoiding full-weight hash passes."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _metadata_path(reference_path: Path) -> Path:
        return reference_path.with_suffix(".json")

    @staticmethod
    def _write_torch_atomic(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def begin_phase(
        self,
        student,
        phase_index: int,
        global_update: int,
        prompt_cursor: int,
    ):
        reference_id = f"phase-{phase_index:02d}-student-step-{global_update}"
        path = self.root / f"phase-{phase_index:02d}-reference.pt"
        metadata_path = self._metadata_path(path)
        if path.exists() or metadata_path.exists():
            if not path.is_file() or not metadata_path.is_file():
                raise RuntimeError(f"incomplete phase-reference pair at {path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata != {
                "phase_index": phase_index,
                "global_update": global_update,
                "reference_id": reference_id,
            }:
                raise RuntimeError(f"phase-reference identity collision at {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            phase_ref = copy.deepcopy(student)
            phase_ref.load_state_dict(payload["state_dict"], strict=True)
            phase_ref.to(next(student.parameters()).device)
            phase_ref.eval().requires_grad_(False)
        else:
            phase_ref = clone_frozen_model(student)
            self._write_torch_atomic(path, {"state_dict": phase_ref.state_dict()})
            train_student.write_json(
                metadata_path,
                {
                    "phase_index": phase_index,
                    "global_update": global_update,
                    "reference_id": reference_id,
                },
            )
        setattr(phase_ref, "_cafd_reference_id", reference_id)
        state = PhaseState(
            phase_index=phase_index,
            phase_local_update=0,
            global_update=global_update,
            prompt_cursor=prompt_cursor,
            phase_reference_path=str(path.resolve()),
            phase_reference_hash=reference_id,
        )
        if _reference_identity(phase_ref) != state.phase_reference_hash:
            raise RuntimeError("new phase reference is not frozen")
        return phase_ref, state

    def save_resume(
        self,
        path,
        *,
        student,
        optimizer,
        scheduler,
        state: PhaseState,
        extra: dict[str, Any] | None = None,
    ) -> None:
        reference_path = Path(state.phase_reference_path)
        metadata = json.loads(self._metadata_path(reference_path).read_text(encoding="utf-8"))
        if metadata["reference_id"] != state.phase_reference_hash:
            raise RuntimeError("persisted phase-reference identity changed")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._write_torch_atomic(
            target,
            {
                "student": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": None if scheduler is None else scheduler.state_dict(),
                "rng": _base_phase_controller._rng_state(),
                "phase": asdict(state),
                "extra": extra or {},
            },
        )

    def load_resume(
        self,
        path,
        *,
        student,
        optimizer,
        scheduler,
        device,
    ):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        student.load_state_dict(payload["student"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload["scheduler"] is not None:
            scheduler.load_state_dict(payload["scheduler"])
        state = PhaseState(**payload["phase"])
        reference_path = Path(state.phase_reference_path)
        metadata = json.loads(self._metadata_path(reference_path).read_text(encoding="utf-8"))
        if metadata["reference_id"] != state.phase_reference_hash:
            raise RuntimeError("resume phase-reference identity changed")
        reference_payload = torch.load(reference_path, map_location="cpu", weights_only=False)
        phase_ref = copy.deepcopy(student)
        phase_ref.load_state_dict(reference_payload["state_dict"], strict=True)
        phase_ref.to(device)
        phase_ref.eval().requires_grad_(False)
        setattr(phase_ref, "_cafd_reference_id", state.phase_reference_hash)
        if _reference_identity(phase_ref) != state.phase_reference_hash:
            raise RuntimeError("restored phase reference is not frozen")
        _base_phase_controller._restore_rng(payload["rng"])
        return state, phase_ref, payload.get("extra", {})


def install_full_run() -> None:
    install_into(train_student)
    train_student.load_experiment_config = _load_full_config
    train_student._teacher_set = _teacher_route
    train_student._generate_rollouts = _teacher_replay_rollouts
    train_student.PhaseController = NoHashPhaseController
    train_student.parameter_sha256 = _reference_identity


if __name__ == "__main__":
    install_full_run()
    train_student.main()
