from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import atomic_json, stable_fraction
from .splits import load_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-stream", type=Path, default=Path("data/frozen/math_training/student_prompt_stream.jsonl"))
    parser.add_argument("--teacher-successes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gold = load_jsonl(args.student_stream)
    teacher = load_jsonl(args.teacher_successes)
    required_rft = 200 * 32
    if len(teacher) < required_rft:
        status = {
            "state": "RFT_DATA_INSUFFICIENT",
            "teacher_success_count": len(teacher),
            "required_without_replacement": required_rft,
        }
        atomic_json(args.output_dir / "status.json", status)
        print(json.dumps(status, indent=2))
        raise SystemExit(3)

    teacher_ordered = sorted(
        teacher,
        key=lambda row: stable_fraction(f"{row['problem_id']}:{row['rollout_slot']}", 42),
    )
    rft = [
        {
            "id": f"teacher:{row['problem_id']}:{row['rollout_slot']}",
            "prompt": row["prompt"],
            "completion": row["completion"],
            "source": "verified_teacher_response",
        }
        for row in teacher_ordered[:required_rft]
    ]
    write_jsonl(args.output_dir / "rft_stream.jsonl", rft)

    capacity_pool = [
        {
            "id": f"gold:{row['id']}",
            "prompt": row["prompt"],
            "completion": row["solution"],
            "source": "legal_training_gold_solution",
        }
        for row in gold
    ] + rft
    capacity_pool = sorted(capacity_pool, key=lambda row: stable_fraction(row["id"], 42))
    capacity_count = 800 * 32
    capacity = [dict(capacity_pool[index % len(capacity_pool)], exposure_index=index) for index in range(capacity_count)]
    write_jsonl(args.output_dir / "capacity_stream.jsonl", capacity)
    status = {
        "state": "COMPLETED",
        "teacher_success_count": len(teacher),
        "rft_exposures": len(rft),
        "rft_sampling": "without_replacement",
        "capacity_exposures": len(capacity),
        "capacity_sources": ["legal_training_gold_solution", "verified_teacher_response"],
    }
    atomic_json(args.output_dir / "status.json", status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
