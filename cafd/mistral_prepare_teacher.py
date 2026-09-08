"""Freeze the selected Teacher seed and the full raw-to-RL acquisition route."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .configuration import load_experiment_config
from .layout import experiment_layout
from .mistral_runtime import validate_config
from .training_common import checkpoint_complete, write_json


def _checked(path: Path) -> str:
    resolved = path.resolve()
    if not checkpoint_complete(resolved):
        raise RuntimeError(f"incomplete Mistral checkpoint: {resolved}")
    if "qwen" in str(resolved).lower():
        raise RuntimeError(f"QWEN_BANNED: {resolved}")
    return str(resolved)


def select_seed(root: Path) -> None:
    config = load_experiment_config(root)
    validate_config(config)
    layout = experiment_layout(root, config)
    gate = json.loads(
        (layout.run_root / "teacher_capacity" / "gate.json").read_text(
            encoding="utf-8"
        )
    )
    if gate["status"] != "passed" or int(gate["development_correct"]) < 52:
        raise RuntimeError(f"Teacher capacity did not pass: {gate}")
    source = Path(_checked(Path(gate["checkpoint"])))
    target = layout.run_root / "teacher_capacity" / "selected"
    if target.is_symlink() or target.exists():
        if target.resolve() != source:
            raise RuntimeError(f"selected Teacher seed changed: {target.resolve()}")
    else:
        os.symlink(source, target, target_is_directory=True)
    write_json(
        layout.state_root / "teacher_seed.json",
        {"status": "frozen", "checkpoint": str(source)},
    )


def freeze_route(root: Path) -> None:
    config = load_experiment_config(root)
    validate_config(config)
    layout = experiment_layout(root, config)
    route_path = layout.teacher / "route.json"
    rl_route = json.loads(route_path.read_text(encoding="utf-8"))
    if rl_route.get("status") != "frozen" or len(rl_route["checkpoints"]) != 6:
        raise RuntimeError(f"invalid RL-only route: {rl_route}")
    backup = layout.teacher / "route_rl_only.json"
    if not backup.exists():
        write_json(backup, rl_route)

    raw = layout.run_root / "teacher_capacity" / "TBase"
    gate = json.loads(
        (layout.run_root / "teacher_capacity" / "gate.json").read_text(
            encoding="utf-8"
        )
    )
    sft = Path(gate["checkpoint"])
    later_rl = rl_route["checkpoints"][2:]
    records = [
        {"step": "raw_instruct", "checkpoint": _checked(raw)},
        {"step": f"sft{gate['selected_step']}", "checkpoint": _checked(sft)},
    ]
    records.extend(
        {
            "step": f"rl{item['step']}",
            "checkpoint": _checked(Path(item["checkpoint"])),
        }
        for item in later_rl
    )
    if len(records) != 6:
        raise RuntimeError(f"full acquisition route is not six checkpoints: {records}")
    route = {
        "status": "frozen",
        "kind": "mistral_full_acquisition_raw_instruct_to_sft_to_rl",
        "checkpoints": records,
        "rl_only_route": str(backup.resolve()),
    }
    write_json(route_path, route)
    write_json(layout.state_root / "full_acquisition_route.json", route)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("select-seed", "freeze-route"), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "select-seed":
        select_seed(root)
    else:
        freeze_route(root)


if __name__ == "__main__":
    main()
