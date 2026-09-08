"""Frozen CAFD-MPC split preparation and metadata-only checkpoint preflight.

No CUDA, model loading, weight hashing, optimizer, generation, or test evaluation.
The official test source is streamed for top-level IDs only; its labels are never
used or persisted. The test has historical evaluations and is NOT untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

FAMILIES = ("contains_count", "contains_ordered", "contains_substring")
SOURCE = Path("data/cafd/generalization_v10")
OUTPUT = Path("data/cafd/mpc_v1")
TEACHER_RUN = Path("runs/cafd/experiments/mistral_cafd_disjoint_v7")
ROUTE = TEACHER_RUN / "teacher/route.json"
TEACHER_SPLIT = TEACHER_RUN / "teacher_data_split.json"
STUDENT_BASE = TEACHER_RUN / "student_base/S0"
MODEL_IDENTITIES = {
    "student": {
        "id": "mistralai/Ministral-3-3B-Instruct-2512-BF16",
        "revision": "b6d637bef2393152b3da2b2fde72eecdee30557e",
    },
    "teacher": {
        "id": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
        "revision": "f6fae9795746f63c9be8344932f01275f3c63734",
    },
}


def _inside(root: Path, path: Path | str) -> Path:
    path = Path(path)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path escapes CAFD root: {path}")
    return resolved


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _ids(rows: list[dict[str, Any]], name: str) -> list[str]:
    result = [row["id"] for row in rows]
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name}: invalid IDs")
    if len(result) != len(set(result)):
        raise ValueError(f"{name}: duplicate or empty IDs")
    return result


def _id_list(value: Any, name: str, expected: int) -> list[str]:
    if not isinstance(value, list) or len(value) != expected:
        raise ValueError(f"{name}: expected {expected} IDs")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name}: invalid IDs")
    if len(set(value)) != expected:
        raise ValueError(f"{name}: duplicate IDs")
    return value


def _stream_ids_only(path: Path, name: str, count: int) -> list[str]:
    # Retain only the top-level ID. Never return, score or persist test payload.
    ids = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.append(json.loads(line)["id"])
    return _id_list(ids, name, count)


def _metadata(root: Path, path: Path | str) -> dict[str, Any]:
    path = _inside(root, path)
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    if stat.st_size <= 0:
        raise ValueError(f"empty required file: {path}")
    return {
        "path": str(path.relative_to(root)),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def checkpoint_metadata(root: Path, path: Path | str) -> dict[str, Any]:
    """Validate declared shards by index and stat; never read/hash model weights."""
    root = root.resolve()
    checkpoint = _inside(root, path)
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    config_path = checkpoint / "config.json"
    config = _json(config_path)
    if config.get("model_type") not in {"mistral3", "ministral3"}:
        raise ValueError(f"non-Mistral checkpoint: {checkpoint}")
    text_config = config.get("text_config", config)
    if text_config.get("vocab_size") != 131072:
        raise ValueError(f"wrong Mistral vocabulary: {checkpoint}")
    if "qwen" in str(config).lower():
        raise ValueError("QWEN_BANNED")
    files = [
        _metadata(root, config_path),
        _metadata(root, checkpoint / "tokenizer_config.json"),
        _metadata(root, checkpoint / "tokenizer.json"),
    ]
    indices = [checkpoint / name for name in
               ("model.safetensors.index.json", "pytorch_model.bin.index.json")
               if (checkpoint / name).exists()]
    if len(indices) > 1:
        raise ValueError(f"ambiguous model indices: {checkpoint}")
    if indices:
        index = _json(indices[0])
        mapping = index.get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"empty weight_map: {indices[0]}")
        if any(not isinstance(name, str) for name in mapping.values()):
            raise ValueError(f"unsafe shard path: {indices[0]}")
        shards = set(mapping.values())
        if any(Path(name).name != name
               or name in {".", ".."} for name in shards):
            raise ValueError(f"unsafe shard path: {indices[0]}")
        files.append(_metadata(root, indices[0]))
    else:
        shards = {name for name in ("model.safetensors", "pytorch_model.bin")
                  if (checkpoint / name).exists()}
        if len(shards) != 1:
            raise ValueError(f"missing or ambiguous model weights: {checkpoint}")
    for shard in sorted(shards):
        entry = _metadata(root, checkpoint / shard)
        if entry["size_bytes"] <= 1_000_000:
            raise ValueError(f"incomplete model shard: {checkpoint / shard}")
        files.append(entry)
    return {
        "checkpoint": str(checkpoint),
        "model_config": config,
        "files": files,
        "weight_validation": "index_and_path_size_mtime_metadata_only",
        "weight_hash_computed": False,
    }


def preflight_route(root: Path) -> dict[str, Any]:
    root = root.resolve()
    route = _json(_inside(root, ROUTE))
    if (route.get("status") != "frozen" or
            route.get("kind") != "mistral_full_acquisition_raw_instruct_to_sft_to_rl"):
        raise ValueError("expected frozen real-order Mistral SFT+RL route")
    checkpoints = route.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 6:
        raise ValueError("expected six real Teacher checkpoints")
    expected_steps = ["raw_instruct", "sft125", "rl40", "rl60", "rl80", "rl100"]
    if [item.get("step") for item in checkpoints] != expected_steps:
        raise ValueError("Teacher acquisition order changed")
    records = [dict(step=item["step"], **checkpoint_metadata(root, item["checkpoint"]))
               for item in checkpoints]
    if len({item["checkpoint"] for item in records}) != 6:
        raise ValueError("Teacher checkpoints must be distinct paths")
    return {
        "route_file": _metadata(root, ROUTE),
        "route_kind": route["kind"],
        "checkpoints": records,
        "student_base": checkpoint_metadata(root, STUDENT_BASE),
        "models": MODEL_IDENTITIES,
        "token_id_equality_checked_here": False,
        "runtime_tokenizer_pair_check_required": True,
    }


def _publish_unchanged(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"refusing to replace frozen MPC file: {path}")
        return
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _pretty(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def prepare(root: Path, seed: int = 2027) -> dict[str, Any]:
    """Publish a new, deterministic 602/12/64 split; never modify old splits."""
    root = root.resolve()
    nested = _json(_inside(root, SOURCE / "manifest.json"))
    if nested.get("status") != "frozen":
        raise ValueError("source nested split is not frozen")
    fit = _rows(_inside(root, SOURCE / "fit.jsonl"))
    development = _rows(_inside(root, SOURCE / "selection.jsonl"))
    fit_ids = _ids(fit, "fit")
    development_ids = _ids(development, "development")
    if fit_ids != _id_list(nested.get("fit_ids"), "fit manifest", 614):
        raise ValueError("fit source differs from frozen ID order")
    if development_ids != _id_list(nested.get("selection_ids"), "selection manifest", 64):
        raise ValueError("selection source differs from frozen ID order")
    if set(fit_ids) & set(development_ids):
        raise ValueError("fit and development overlap")
    if set(row["problem_family"] for row in fit + development) != set(FAMILIES):
        raise ValueError("unexpected problem families")
    rng = random.Random(int(seed))
    control = []
    for family in FAMILIES:
        candidates = [row for row in fit if row["problem_family"] == family]
        if len(candidates) <= 4:
            raise ValueError(f"insufficient optimization/control tasks: {family}")
        rng.shuffle(candidates)
        control.extend(candidates[:4])
    control_ids = _ids(control, "control")
    control_set = set(control_ids)
    optimization = [row for row in fit if row["id"] not in control_set]
    optimization_ids = _ids(optimization, "optimization")
    if len(optimization_ids) != 602 or len(control_ids) != 12:
        raise ValueError("invalid MPC split counts")
    test_source = _inside(root, "data/cafd/test.jsonl")
    test_ids = _stream_ids_only(test_source, "official test", 132)
    confirmation_ids = _stream_ids_only(
        _inside(root, "data/cafd/development.jsonl"), "unused confirmation", 64)
    partitions = {
        "optimization": optimization_ids, "control": control_ids,
        "development": development_ids, "official_test": test_ids,
        "unused_confirmation": confirmation_ids,
    }
    seen = set()
    for name, ids in partitions.items():
        if seen & set(ids):
            raise ValueError(f"split ID overlap: {name}")
        seen.update(ids)
    teacher_split = _json(_inside(root, TEACHER_SPLIT))
    if teacher_split.get("status") != "frozen":
        raise ValueError("Teacher source split is not frozen")
    sft_ids = set(_id_list(teacher_split.get("teacher_sft_ids"), "Teacher SFT", 512))
    rl_ids = set(_id_list(teacher_split.get("teacher_rlvr_ids"), "Teacher RL", 166))
    if sft_ids & rl_ids or sft_ids | rl_ids != set(fit_ids + development_ids):
        raise ValueError("Teacher training ID partition changed")
    teacher_seen_development = sorted(sft_ids & set(development_ids))
    if len(teacher_seen_development) != 44:
        raise ValueError("expected historical Teacher SFT exposure of 44/64 selection tasks")
    route = preflight_route(root)
    rows_by_split = {"optimization": optimization, "control": control,
                     "development": development}
    manifest = {
        "version": "cafd_mpc_v1", "status": "frozen", "seed": int(seed),
        "benchmark": "DELTA Manufactoria-HAS", "metric": "official full-pass/pass@1",
        "models": MODEL_IDENTITIES,
        "source_nested_manifest": _metadata(root, SOURCE / "manifest.json"),
        "source_fit": _metadata(root, SOURCE / "fit.jsonl"),
        "source_selection": _metadata(root, SOURCE / "selection.jsonl"),
        "teacher_split_manifest": _metadata(root, TEACHER_SPLIT),
        "control_tasks_per_family": 4, "control_is_training_side": True,
        "partitions": {
            name: {"count": len(rows), "ids": partitions[name],
                   "path": str(OUTPUT / f"{name}.jsonl"),
                   "family_counts": dict(sorted(Counter(
                       row["problem_family"] for row in rows).items()))}
            for name, rows in rows_by_split.items()
        },
        "official_test": {
            "count": 132, "ids_metadata_path": str(OUTPUT / "test_ids.json"),
            "source_file_metadata": _metadata(root, test_source),
            "frozen_test_access_scope": "problem_ids_only",
            "labels_used": False, "evaluated": False,
            "historically_evaluated": True, "untouched_test_claim": False,
        },
        "unused_confirmation": {
            "source": "data/cafd/development.jsonl", "count": 64,
            "ids": confirmation_ids, "used_for_mpc": False,
        },
        "historical_exposure": {
            "teacher_sft_seen_development_count": len(teacher_seen_development),
            "teacher_sft_seen_development_ids": teacher_seen_development,
            "teacher_rl_seen_development_count": len(rl_ids & set(development_ids)),
            "development_used_by_previous_student_experiments": True,
            "whole_pipeline_held_out_claim": False,
        },
        "limitations": [
            "ID-disjointness is for the new Student optimization/control/development splits.",
            "44/64 development tasks were used by Teacher SFT; the other 20 by Teacher RL.",
            "Development and official test were evaluated in historical experiments.",
            "This preparation reads test problem IDs only, not using labels or evaluating models.",
            "File metadata is not a cryptographic integrity guarantee.",
            "Exact tokenizer and token-ID equality remains a required runtime assertion.",
        ],
        "preflight": route,
    }
    output = _inside(root, OUTPUT)
    output.mkdir(parents=True, exist_ok=True)
    # Lock only the new MPC output directory. Existing splits remain untouched.
    import fcntl
    with (output / ".prepare.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = output / "manifest.json"
        if existing.exists() and _json(existing) != manifest:
            raise RuntimeError("MPC split/route is already frozen differently")
        for split, rows in rows_by_split.items():
            _publish_unchanged(output / f"{split}.jsonl", "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
        _publish_unchanged(output / "test_ids.json", _pretty({
            "ids": test_ids, "count": 132,
            "source": "data/cafd/test.jsonl",
            "access_scope": "problem_ids_only",
            "labels_used": False, "evaluated": False,
            "historically_evaluated": True,
        }))
        _publish_unchanged(existing, _pretty(manifest))
    return manifest


def load_mpc_rows(root: Path, split: str) -> list[dict[str, Any]]:
    if split not in {"optimization", "control", "development"}:
        raise ValueError(f"MPC cannot load undeclared or test split: {split}")
    root = root.resolve()
    manifest = _json(_inside(root, OUTPUT / "manifest.json"))
    if manifest.get("status") != "frozen":
        raise ValueError("MPC splits are not frozen")
    rows = _rows(_inside(root, OUTPUT / f"{split}.jsonl"))
    entry = manifest["partitions"][split]
    if len(rows) != entry["count"] or _ids(rows, split) != entry["ids"]:
        raise ValueError(f"frozen MPC {split} IDs changed")
    families = dict(sorted(Counter(row["problem_family"] for row in rows).items()))
    if families != entry["family_counts"]:
        raise ValueError(f"frozen MPC {split} families changed")
    source_name = "selection.jsonl" if split == "development" else "fit.jsonl"
    metadata_key = "source_selection" if split == "development" else "source_fit"
    if _metadata(root, SOURCE / source_name) != manifest[metadata_key]:
        raise ValueError("frozen source metadata changed")
    source = {row["id"]: row for row in _rows(_inside(root, SOURCE / source_name))}
    if rows != [source[identifier] for identifier in entry["ids"]]:
        raise ValueError(f"frozen MPC {split} payload changed")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    result = prepare(args.root, args.seed)
    print(json.dumps({
        "status": result["status"],
        "counts": {key: value["count"] for key, value in result["partitions"].items()},
        "teacher_sft_seen_development": result["historical_exposure"]["teacher_sft_seen_development_count"],
        "test_access_scope": result["official_test"]["frozen_test_access_scope"],
        "route_checkpoints": len(result["preflight"]["checkpoints"]),
    }, sort_keys=True))
