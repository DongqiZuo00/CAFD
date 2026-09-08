from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import atomic_json, stable_fraction
from .splits import load_jsonl, write_jsonl
from .verifiers import verify_math


def canonical_completion(solution: str) -> str:
    return solution if "\\boxed{" in solution else f"\\boxed{{{solution}}}"


def prepare_oracle_rows(rows: list[dict], seed: int, exposure_count: int) -> tuple[list[dict], list[dict]]:
    verified = []
    failures = []
    for row in rows:
        completion = canonical_completion(str(row["solution"]))
        result = verify_math(completion, str(row["solution"]))
        record = {
            "id": f"canonical:{row['id']}",
            "problem_id": row["id"],
            "stream_position": int(row["stream_position"]),
            "prompt": row["prompt"],
            "completion": completion,
            "source": "verified_canonical_training_solution",
            "verifier_output": result.as_dict(),
        }
        if result.correct:
            verified.append(record)
        else:
            failures.append(record)
    ordered = sorted(verified, key=lambda row: stable_fraction(str(row["problem_id"]), seed))
    exposures = [
        {**ordered[index % len(ordered)], "exposure_index": index}
        for index in range(exposure_count)
    ] if ordered else []
    return exposures, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--failure-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-updates", type=int, default=800)
    parser.add_argument("--sequences-per-update", type=int, default=32)
    args = parser.parse_args()
    rows = load_jsonl(args.input)
    exposures, failures = prepare_oracle_rows(
        rows, args.seed, args.max_updates * args.sequences_per_update
    )
    write_jsonl(args.output, exposures)
    write_jsonl(args.failure_output, failures)
    status = {
        "state": "COMPLETED" if not failures else "CANONICAL_SOLUTIONS_REQUIRE_FALLBACK",
        "training_prompts": len(rows),
        "verified_canonical_prompts": len(rows) - len(failures),
        "canonical_failures": len(failures),
        "cached_responses_per_prompt": 1,
        "exposures": len(exposures),
        "max_updates": args.max_updates,
        "sequences_per_update": args.sequences_per_update,
        "seed": args.seed,
    }
    atomic_json(args.status, status)
    print(json.dumps(status, indent=2))
    if failures:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
