from __future__ import annotations

import json
from pathlib import Path

from .io import ARTIFACT_ROOT, ROOT, atomic_json, load_yaml, write_csv
from .scheduler import initialize


CSV_SCHEMAS = {
    "overlap_audit.csv": ["training_id", "training_source", "target_id", "target_source", "match_type", "similarity"],
    "teacher_curves.csv": ["setting", "method", "step", "split", "metric", "score", "run_id"],
    "student_curves.csv": ["setting", "method", "step", "split", "metric", "score", "run_id"],
    "final_scores.csv": ["setting", "method", "split", "metric", "score", "correct", "total", "run_id"],
    "confidence_intervals.csv": ["setting", "method", "metric", "score", "lower_95", "upper_95", "bootstrap_seed"],
    "costs.csv": [
        "setting", "method", "gpu_hours", "wall_seconds", "student_generated_tokens",
        "teacher_scored_tokens", "verifier_calls", "peak_memory_bytes", "run_id"
    ],
}


def main() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    for directory in ("raw_predictions", "verifier_outputs", "checkpoints", "frozen_configs", "tests"):
        (ARTIFACT_ROOT / directory).mkdir(parents=True, exist_ok=True)
    for name, fields in CSV_SCHEMAS.items():
        path = ARTIFACT_ROOT / name
        if not path.exists():
            write_csv(path, [], fields)
    backbones = load_yaml("backbones.yaml")
    models = {
        "schema_version": 1,
        "frozen_at_initialization": True,
        "pairs": backbones["pairs"],
        "note": "null revisions are access-blocked and are not silently resolved or replaced",
    }
    model_manifest = ARTIFACT_ROOT / "model_manifest.json"
    if not model_manifest.exists():
        atomic_json(model_manifest, models)
    initialize()
    print(json.dumps({"artifact_root": str(ARTIFACT_ROOT), "workspace": str(ROOT)}, indent=2))


if __name__ == "__main__":
    main()
