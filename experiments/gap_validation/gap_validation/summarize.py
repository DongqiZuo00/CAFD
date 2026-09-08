from __future__ import annotations

import csv
import json
from pathlib import Path

from .io import ARTIFACT_ROOT, atomic_json


BASELINES = ["T0", "S0", "Direct_RLVR", "Teacher_RFT", "Endpoint_GKD", "Progressive_GKD"]


def main() -> None:
    score_path = ARTIFACT_ROOT / "final_scores.csv"
    if not score_path.exists():
        result = {"status": "NO_FINAL_RESULTS", "gap_established": False}
        atomic_json(ARTIFACT_ROOT / "gap_status.json", result)
        print(json.dumps(result, indent=2))
        return
    with score_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, dict[str, float]] = {}
    for row in rows:
        if not row.get("score"):
            continue
        grouped.setdefault(row["setting"], {})[row["method"]] = float(row["score"])
    settings = {}
    for setting, values in grouped.items():
        required = BASELINES + ["T400", "Capacity_Control"]
        if not all(name in values for name in required):
            settings[setting] = {"status": "INCOMPLETE"}
            continue
        lower = max(values[name] for name in BASELINES)
        upper = min(values["T400"], values["Capacity_Control"])
        settings[setting] = {
            "status": "SINGLE_SEED_GAP_COMPATIBLE" if lower < upper else "NOT_GAP_COMPATIBLE",
            "lower_endpoint": lower,
            "upper_endpoint": upper,
            "compatible_interval": f"({lower}, {upper}]" if lower < upper else None,
            "gap_established": False,
        }
    result = {"status": "SINGLE_SEED_CANDIDATE_GAP", "gap_established": False, "settings": settings}
    atomic_json(ARTIFACT_ROOT / "gap_status.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

