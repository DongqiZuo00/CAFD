"""Immutable phase-reference and exact-resume state for CAFD-v1."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def parameter_sha256(model: nn.Module) -> str:
    return state_dict_sha256(model.state_dict())


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def clone_frozen_model(model: nn.Module) -> nn.Module:
    clone = copy.deepcopy(model)
    clone.eval()
    clone.requires_grad_(False)
    for source, target in zip(model.parameters(), clone.parameters(), strict=True):
        if source.data_ptr() == target.data_ptr():
            raise RuntimeError("phase reference shares parameter storage with trainable Student")
    return clone


@dataclass
class PhaseState:
    phase_index: int
    phase_local_update: int
    global_update: int
    prompt_cursor: int
    phase_reference_path: str
    phase_reference_hash: str


class PhaseController:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def begin_phase(self, student: nn.Module, phase_index: int, global_update: int, prompt_cursor: int) -> tuple[nn.Module, PhaseState]:
        phase_ref = clone_frozen_model(student)
        digest = parameter_sha256(phase_ref)
        path = self.root / f"phase-{phase_index:02d}-reference.pt"
        existing = None
        if path.exists():
            try:
                existing = torch.load(path, map_location="cpu", weights_only=False)
            except (EOFError, OSError, RuntimeError, ValueError, pickle.UnpicklingError):
                aborted = path.with_name(
                    f"{path.name}.aborted-{os.getpid()}-{time.time_ns()}"
                )
                os.replace(path, aborted)
        if existing is not None and existing.get("sha256") != digest:
            raise RuntimeError(f"immutable phase-reference collision at {path}")
        if existing is None:
            temporary = path.with_name(
                f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
            )
            with temporary.open("wb") as handle:
                torch.save(
                    {"state_dict": phase_ref.state_dict(), "sha256": digest},
                    handle,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        state = PhaseState(
            phase_index=phase_index,
            phase_local_update=0,
            global_update=global_update,
            prompt_cursor=prompt_cursor,
            phase_reference_path=str(path.resolve()),
            phase_reference_hash=digest,
        )
        return phase_ref, state

    @staticmethod
    def load_phase_reference(template: nn.Module, state: PhaseState, device: torch.device | str) -> nn.Module:
        payload = torch.load(state.phase_reference_path, map_location="cpu", weights_only=False)
        if payload["sha256"] != state.phase_reference_hash:
            raise RuntimeError("saved phase-reference hash does not match resume state")
        phase_ref = copy.deepcopy(template)
        phase_ref.load_state_dict(payload["state_dict"], strict=True)
        phase_ref.to(device)
        phase_ref.eval().requires_grad_(False)
        if parameter_sha256(phase_ref) != state.phase_reference_hash:
            raise RuntimeError("restored phase-reference weights changed")
        return phase_ref

    @staticmethod
    def _rng_state() -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng(state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and "torch_cuda" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    def save_resume(
        self,
        path: str | os.PathLike[str],
        *,
        student: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        state: PhaseState,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if parameter_sha256(self._phase_template_for_hash(student, state)) != state.phase_reference_hash:
            raise RuntimeError("phase reference changed before resume save")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(
            {
                "student": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": None if scheduler is None else scheduler.state_dict(),
                "rng": self._rng_state(),
                "phase": asdict(state),
                "extra": extra or {},
            },
            temporary,
        )
        os.replace(temporary, target)

    @staticmethod
    def _phase_template_for_hash(student: nn.Module, state: PhaseState) -> nn.Module:
        payload = torch.load(state.phase_reference_path, map_location="cpu", weights_only=False)
        if state_dict_sha256(payload["state_dict"]) != state.phase_reference_hash:
            raise RuntimeError("persisted phase reference changed")
        # Return a lightweight proxy exposing state_dict for parameter_sha256.
        class _Proxy(nn.Module):
            def state_dict(self, *args, **kwargs):  # type: ignore[override]
                return payload["state_dict"]

        return _Proxy()

    def load_resume(
        self,
        path: str | os.PathLike[str],
        *,
        student: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device | str,
    ) -> tuple[PhaseState, nn.Module, dict[str, Any]]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        student.load_state_dict(payload["student"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload["scheduler"] is not None:
            scheduler.load_state_dict(payload["scheduler"])
        state = PhaseState(**payload["phase"])
        phase_ref = self.load_phase_reference(student, state, device)
        self._restore_rng(payload["rng"])
        return state, phase_ref, payload.get("extra", {})

    @staticmethod
    def write_status(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, target)
