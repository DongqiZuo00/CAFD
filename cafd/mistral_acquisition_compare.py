"""Freeze the development-only mechanism comparison between CAFD and Progressive GKD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .training_common import write_json


RUNS = {
    "CAFD": ("mistral_cafd_gen_v10_mixed", "cafd"),
    "Progressive GKD": ("mistral_progressive_gkd_observation_v11", "progressive"),
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_at_least(records: list[dict[str, Any]], correct: int) -> int | None:
    for row in sorted(records, key=lambda item: int(item["step"])):
        if int(row["correct"]) >= correct:
            return int(row["step"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    observations = {}
    for label, (run_id, _) in RUNS.items():
        record = _load(
            root / "artifacts" / "cafd" / "experiments" / run_id
            / "acquisition_observation.json"
        )
        if record.get("status") != "complete" or record.get("frozen_test_accessed") is not False:
            raise RuntimeError(f"incomplete or contaminated observation: {label}")
        observations[label] = record

    cafd = observations["CAFD"]
    gkd = observations["Progressive GKD"]
    cafd_selected = cafd["selected_record"]
    gkd_selected = gkd["selected_record"]
    family_deltas = {
        family: (
            int(cafd_selected["by_family"][family]["correct"])
            - int(gkd_selected["by_family"][family]["correct"])
        )
        for family in ("contains_count", "contains_ordered", "contains_substring")
    }
    same_step_deltas = []
    gkd_by_step = {int(row["step"]): row for row in gkd["records"]}
    for row in cafd["records"]:
        step = int(row["step"])
        same_step_deltas.append({
            "step": step,
            "cafd_correct": int(row["correct"]),
            "progressive_gkd_correct": int(gkd_by_step[step]["correct"]),
            "delta_correct": int(row["correct"]) - int(gkd_by_step[step]["correct"]),
        })
    payload = {
        "status": "development_only_mechanism_observation_frozen",
        "primary_question": (
            "Does acquisition-delta transfer improve over absolute-checkpoint "
            "distillation under identical mixed rollout support?"
        ),
        "controlled_factors": [
            "Student and Teacher backbones",
            "Teacher acquisition route",
            "614-example fit split and 64-example checkpoint-selection split",
            "64-example development observation split",
            "Teacher/Student rollout-support schedule",
            "prompt stream, optimizer, 200-update budget, and milestones",
        ],
        "only_changed_factor": (
            "CAFD uses phase_ref + next_teacher - previous_teacher; "
            "Progressive GKD uses next_teacher absolute logits"
        ),
        "CAFD": cafd,
        "Progressive GKD": gkd,
        "selected_confirmation_delta_correct": (
            int(cafd_selected["correct"]) - int(gkd_selected["correct"])
        ),
        "normalized_accuracy_auc_delta": (
            float(cafd["normalized_accuracy_auc"])
            - float(gkd["normalized_accuracy_auc"])
        ),
        "family_deltas_at_selected_checkpoints": family_deltas,
        "same_step_deltas": same_step_deltas,
        "first_step_at_32_of_64": {
            label: _first_at_least(record["records"], 32)
            for label, record in observations.items()
        },
        "retention_from_peak_correct": {
            label: int(record["retention_from_peak_correct"])
            for label, record in observations.items()
        },
        "frozen_test_accessed": False,
    }
    output = root / "artifacts" / "cafd" / "acquisition_observation_v11.json"
    if output.exists() and _load(output) != payload:
        raise RuntimeError("mechanism observation already frozen differently")
    write_json(output, payload)


if __name__ == "__main__":
    main()
