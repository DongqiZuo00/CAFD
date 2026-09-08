"""Compare preregistered v10 controls without opening the frozen test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .training_common import write_json


RUNS = (
    "mistral_cafd_gen_v10_teacher",
    "mistral_cafd_gen_v10_mixed",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rank(record: dict[str, Any]) -> tuple[float, float, float, int]:
    confirmation = record["confirmation"]
    family_floor = min(
        float(score["accuracy"]) for score in record["by_family"].values()
    )
    gap = abs(int(record["selection_minus_confirmation"]))
    support_tiebreak = int(record["rollout_support"] == "next_teacher_replay")
    return (
        float(confirmation["accuracy"]),
        family_floor,
        -float(gap),
        support_tiebreak,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    records = {
        run: _load(
            root / "runs" / "cafd" / "experiments" / run
            / "cafd" / "confirmation.json"
        )
        for run in RUNS
    }
    if any(record.get("frozen_test_accessed") is not False for record in records.values()):
        raise RuntimeError("v10 diagnostic touched frozen test")
    winner = max(RUNS, key=lambda run: _rank(records[run]))
    control = records[RUNS[0]]
    mixed = records[RUNS[1]]
    mixed_delta = (
        int(mixed["confirmation"]["correct"])
        - int(control["confirmation"]["correct"])
    )
    payload = {
        "status": "development_only_decision_frozen",
        "selection_rule": [
            "higher confirmation full-pass",
            "higher minimum problem-family accuracy",
            "smaller absolute selection-confirmation gap",
            "prefer simpler pure Teacher replay on exact tie",
        ],
        "winner": winner,
        "winner_support": records[winner]["rollout_support"],
        "mixed_minus_teacher_confirmation_correct": mixed_delta,
        "teacher_replay": control,
        "mixed_support": mixed,
        "frozen_test_accessed": False,
        "interpretation": (
            "teacher-prefix exposure mismatch supported"
            if mixed_delta > 0
            else "teacher-prefix exposure mismatch not supported by this ablation"
        ),
    }
    output = root / "artifacts" / "cafd" / "generalization_v10" / "decision.json"
    if output.exists() and _load(output) != payload:
        raise RuntimeError("v10 development decision is already frozen differently")
    write_json(output, payload)


if __name__ == "__main__":
    main()
