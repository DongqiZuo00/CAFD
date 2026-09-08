from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset

from .io import ARTIFACT_ROOT, ROOT, atomic_json, load_yaml, sha256_file, stable_fraction, write_csv
from .overlap import find_overlaps, match_dicts, normalize_prompt
from .splits import load_jsonl, stable_split, stratified_holdout, write_jsonl


FROZEN_ROOT = ROOT / "data" / "frozen"


def prepare_olymmath() -> dict[str, Any]:
    config = load_yaml("benchmarks.yaml")["primary"]["olymmath_english_easy"]
    source = ROOT / config["path"]
    records = load_jsonl(source)
    development, final_test = stratified_holdout(
        records,
        id_field="unique_id",
        stratum_field="subject",
        seed=config["split"]["seed"],
        development_fraction=config["split"]["development_fraction"],
    )
    out = FROZEN_ROOT / "olymmath"
    write_jsonl(out / "development.jsonl", development)
    write_jsonl(out / "final_test.jsonl", final_test)
    hard_source = ROOT / "vendor/OlymMATH/data/OlymMATH-EN-HARD.jsonl"
    hard = load_jsonl(hard_source)
    write_jsonl(out / "hard_untouched.jsonl", hard)
    manifest = {
        "source_revision": config["revision"],
        "source_sha256": sha256_file(source),
        "development_count": len(development),
        "final_test_count": len(final_test),
        "hard_count": len(hard),
        "development_ids": sorted(str(row["unique_id"]) for row in development),
        "final_test_ids": sorted(str(row["unique_id"]) for row in final_test),
        "hard_ids": sorted(str(row["unique_id"]) for row in hard),
        "subject_distribution": {
            "development": dict(Counter(str(row.get("subject")) for row in development)),
            "final_test": dict(Counter(str(row.get("subject")) for row in final_test)),
        },
    }
    atomic_json(out / "manifest.json", manifest)
    return manifest


def _math_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row["extra_info"]["index"]),
        "source": str(row.get("data_source", "math_dapo")),
        "prompt": str(row["prompt"]),
        "solution": str(row["solution"]),
        "reward_style": str(row.get("reward_model", {}).get("style", "")),
    }


def prepare_math_training() -> dict[str, Any]:
    config = load_yaml("data_protocol.yaml")["math"]
    dataset = load_dataset(
        config["dataset"],
        config["config"],
        split=config["source_split"],
        revision=config["revision"],
    )
    unique: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for raw in dataset:
        row = _math_record(raw)
        normalized = normalize_prompt(row["prompt"])
        if normalized in unique:
            duplicate_ids.append(row["id"])
            continue
        unique[normalized] = row
    records = list(unique.values())
    olym = FROZEN_ROOT / "olymmath"
    targets: list[dict[str, Any]] = []
    for split in ("development", "final_test", "hard_untouched"):
        for row in load_jsonl(olym / f"{split}.jsonl"):
            targets.append(
                {
                    "id": str(row["unique_id"]),
                    "source": f"olymmath_{split}",
                    "prompt": str(row["problem"]),
                    "solution": str(row["answer"]),
                }
            )
    matches = find_overlaps(records, targets)
    removed_ids = {match.training_id for match in matches}
    cleaned = [row for row in records if row["id"] not in removed_ids]
    train, validation = stable_split(cleaned, "id", seed=42, validation_fraction=0.10)
    out = FROZEN_ROOT / "math_training"
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "internal_validation.jsonl", validation)
    ordered = sorted(train, key=lambda row: stable_fraction(str(row["id"]), 42))
    if len(ordered) < 3200:
        raise RuntimeError("math training split cannot supply the frozen 400-update prompt stream")
    teacher_stream = [dict(row, stream_position=index) for index, row in enumerate(ordered[:3200])]
    student_stream = [dict(row, stream_position=index) for index, row in enumerate(ordered[:1600])]
    write_jsonl(out / "teacher_prompt_stream.jsonl", teacher_stream)
    write_jsonl(out / "student_prompt_stream.jsonl", student_stream)
    manifest = {
        "dataset": config["dataset"],
        "revision": config["revision"],
        "config": config["config"],
        "before_deduplication": len(dataset),
        "duplicates_removed": len(duplicate_ids),
        "overlap_removed": len(removed_ids),
        "after_filtering": len(cleaned),
        "train_count": len(train),
        "internal_validation_count": len(validation),
        "teacher_prompt_stream_count": len(teacher_stream),
        "student_prompt_stream_count": len(student_stream),
        "prompt_stream_order": "ascending_sha256(seed=42, stable_problem_id)",
        "duplicate_ids": sorted(duplicate_ids),
        "removed_ids": sorted(removed_ids),
    }
    atomic_json(out / "manifest.json", manifest)
    overlap_rows = match_dicts(matches)
    atomic_json(out / "overlap.json", overlap_rows)
    write_csv(
        ARTIFACT_ROOT / "overlap_audit.csv",
        overlap_rows,
        ["training_id", "training_source", "target_id", "target_source", "match_type", "similarity"],
    )
    return manifest


def _code_record(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    return {
        "id": str(row["problem_id"]),
        "source": str(row.get("source", "")),
        "prompt": str(row["problem"]),
        "solution": str(row["gold_standard_solution"]),
        "url": str(metadata.get("problem_url") or ""),
        "verification_info": row["verification_info"],
    }


def prepare_code_training() -> dict[str, Any]:
    config = load_yaml("data_protocol.yaml")["code"]
    dataset = load_dataset(
        config["dataset"],
        config["config"],
        split=config["source_split"],
        revision=config["revision"],
    )
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for raw in dataset:
        info = raw.get("verification_info") or {}
        reason = None
        if raw.get("task_type") != "verifiable_code":
            reason = "task_type"
        elif str(info.get("language", "")).lower() != "python":
            reason = "language"
        elif not info.get("test_cases"):
            reason = "missing_tests"
        elif float(raw.get("test_reward") or 0.0) != 1.0:
            reason = "reference_not_verified"
        if reason:
            rejected.append({"id": str(raw.get("problem_id")), "reason": reason})
        else:
            accepted.append(_code_record(raw))
    train, validation = stable_split(accepted, "id", seed=42, validation_fraction=0.05)
    out = FROZEN_ROOT / "code_training"
    write_jsonl(out / "pre_overlap_train.jsonl", train)
    write_jsonl(out / "pre_overlap_internal_validation.jsonl", validation)
    manifest = {
        "dataset": config["dataset"],
        "revision": config["revision"],
        "before_filtering": len(dataset),
        "accepted_before_benchmark_overlap": len(accepted),
        "rejected": rejected,
        "pre_overlap_train_count": len(train),
        "pre_overlap_internal_validation_count": len(validation),
        "status": "DATA_CONTRACT_BLOCKED",
        "blocker": "LiveCodeBench Pro gated prompts unavailable; final overlap audit cannot run",
    }
    atomic_json(out / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("olymmath", "math-training", "code-training", "all"))
    args = parser.parse_args()
    results: dict[str, Any] = {}
    if args.action in ("olymmath", "all"):
        results["olymmath"] = prepare_olymmath()
    if args.action in ("math-training", "all"):
        if not (FROZEN_ROOT / "olymmath" / "manifest.json").exists():
            prepare_olymmath()
        results["math_training"] = prepare_math_training()
    if args.action in ("code-training", "all"):
        results["code_training"] = prepare_code_training()
    current = ARTIFACT_ROOT / "dataset_manifest.json"
    atomic_json(current, results)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
