"""Versioned experiment paths that preserve every earlier CAFD run."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentLayout:
    root: Path
    run_id: str

    @property
    def run_root(self) -> Path:
        return self.root / "runs" / "cafd" / "experiments" / self.run_id

    @property
    def artifact_root(self) -> Path:
        return self.root / "artifacts" / "cafd" / "experiments" / self.run_id

    @property
    def state_root(self) -> Path:
        return self.root / "state" / "cafd" / "experiments" / self.run_id

    @property
    def capacity(self) -> Path:
        return self.run_root / "capacity"

    @property
    def student_base(self) -> Path:
        return self.run_root / "student_base" / "S0"

    @property
    def teacher(self) -> Path:
        return self.run_root / "teacher"

    def method(self, name: str) -> Path:
        if name not in {"direct", "endpoint", "progressive", "cafd"}:
            raise ValueError(name)
        return self.run_root / name

    @property
    def final(self) -> Path:
        return self.run_root / "final"

    @property
    def frozen_selection(self) -> Path:
        return self.run_root / "frozen_selection.json"


def experiment_layout(root: Path, config: dict[str, Any]) -> ExperimentLayout:
    run_id = str(config["experiment"]["run_id"])
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,63}", run_id):
        raise ValueError(f"unsafe experiment run_id: {run_id!r}")
    return ExperimentLayout(root=root.resolve(), run_id=run_id)
