from __future__ import annotations

import argparse
import json
import os
from itertools import groupby
from pathlib import Path

from .io import atomic_json
from .splits import load_jsonl


def select_candidate_prefix(rows: list[dict], required_successes: int) -> list[dict]:
    if required_successes < 1:
        raise ValueError("required_successes must be positive")
    unique: dict[tuple[str, int], dict] = {}
    for row in rows:
        key = (str(row["problem_id"]), int(row["rollout_slot"]))
        if key in unique:
            raise ValueError(f"duplicate teacher response: {key}")
        unique[key] = row
    ordered = sorted(
        unique.values(),
        key=lambda row: (
            int(row["stream_position"]),
            str(row["problem_id"]),
            int(row["rollout_slot"]),
        ),
    )
    selected: list[dict] = []
    success_count = 0
    for _, prompt_rows in groupby(ordered, key=lambda row: int(row["stream_position"])):
        batch = list(prompt_rows)
        selected.extend(batch)
        success_count += sum(bool(row.get("correct")) for row in batch)
        if success_count >= required_successes:
            break
    return selected


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--required-successes", type=int, default=6400)
    args = parser.parse_args()

    shard_files = sorted((args.input_dir / "shards").glob("*/all_candidates.jsonl"))
    if not shard_files:
        raise FileNotFoundError(f"no shard candidates found under {args.input_dir / 'shards'}")
    candidates: list[dict] = []
    for path in shard_files:
        candidates.extend(load_jsonl(path))
    selected = select_candidate_prefix(candidates, args.required_successes)
    successes = [row for row in selected if bool(row.get("correct"))]
    atomic_jsonl(args.input_dir / "all_candidates.jsonl", selected)
    atomic_jsonl(args.input_dir / "teacher_successes.jsonl", successes)

    prompt_runtimes: dict[str, float] = {}
    for row in selected:
        problem_id = str(row["problem_id"])
        prompt_runtimes[problem_id] = max(
            prompt_runtimes.get(problem_id, 0.0),
            float(row.get("runtime_seconds_per_prompt_batch", 0.0)),
        )
    state = "COMPLETED" if len(successes) >= args.required_successes else "RFT_DATA_INSUFFICIENT"
    status = {
        "state": state,
        "num_shards": len(shard_files),
        "completed_prompts": len({str(row["problem_id"]) for row in selected}),
        "candidate_responses": len(selected),
        "successful_responses": len(successes),
        "required_successes": args.required_successes,
        "generated_tokens": sum(int(row.get("generated_tokens", 0)) for row in selected),
        "verifier_calls": len(selected),
        "aggregate_generation_seconds": sum(prompt_runtimes.values()),
    }
    atomic_json(args.input_dir / "status.json", status)
    print(json.dumps(status, indent=2))
    if state != "COMPLETED":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
