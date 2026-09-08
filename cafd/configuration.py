"""Safe experiment-config selection with a frozen default."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path("configs/cafd/manufactoria_has.yaml")


def load_experiment_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    requested = Path(os.environ.get("CAFD_CONFIG", str(DEFAULT_CONFIG)))
    path = requested if requested.is_absolute() else root / requested
    path = path.resolve()
    allowed = (root / "configs" / "cafd").resolve()
    if path.parent != allowed:
        raise RuntimeError(f"CAFD config must be directly inside {allowed}: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))
