"""Recovery-only guards for the interrupted formal v4b Student matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .configuration import load_experiment_config
from .layout import ExperimentLayout, experiment_layout
from .training_common import write_json


RUN_ID = "chat_hier_v4b_sft25"
MILESTONES = [0, 40, 80, 120, 160, 200]
METHOD_LABELS = {
    "direct": "Direct RLVR",
    "endpoint": "Endpoint OPD",
    "progressive": "Progressive GKD",
    "cafd": "CAFD-v1",
}
FINAL_CONDITIONS = [
    "teacher_base",
    "teacher_rlvr",
    "student_base",
    "capacity",
    "direct",
    "endpoint",
    "progressive",
    "cafd",
]


def _require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_complete(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    tokenizer_present = (path / "tokenizer.json").is_file() or (
        path / "tokenizer_config.json"
    ).is_file()
    if not tokenizer_present:
        return False
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / index_name
        if not index_path.is_file():
            continue
        index = _load_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        shards = {path / str(name) for name in weight_map.values()}
        return all(shard.is_file() and shard.stat().st_size > 1_000_000 for shard in shards)
    weights = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
    return len(weights) == 1 and weights[0].stat().st_size > 1_000_000


def _resume_update(path: Path, method: str) -> int:
    import torch

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if method == "cafd":
        update = int(payload["phase"]["global_update"])
    else:
        update = int(payload["update"])
    del payload
    return update


def _protocol(root: Path) -> tuple[dict[str, Any], ExperimentLayout]:
    config = load_experiment_config(root)
    _require(config["protocol"] == "compute_bounded_v4_sft25_warmstart", "wrong protocol")
    _require(config["experiment"]["run_id"] == RUN_ID, "wrong run_id")
    _require(config["seed"] == 2027, "wrong seed")
    _require(config["generation"]["max_new_tokens"] == 2048, "wrong generation cap")
    _require(
        config["generation"]["prompt_renderer"] == "qwen3_instruct_no_thinking",
        "wrong prompt renderer",
    )
    _require(config["reward"]["mode"] == "hierarchical", "wrong reward mode")
    _require(config["student"]["updates"] == 200, "wrong Student budget")
    _require(config["student"]["prompts_per_update"] == 4, "wrong prompt count")
    _require(config["student"]["rollouts_per_prompt"] == 8, "wrong rollout count")
    _require(config["student"]["milestones"] == MILESTONES, "wrong milestones")
    _require(config["cafd"]["phase_updates"] == 40, "wrong CAFD phase length")
    _require(config["cafd"]["gamma"] == 1.0, "wrong CAFD gamma")
    _require(config["cafd"]["temperature"] == 1.0, "wrong CAFD temperature")
    return config, experiment_layout(root, config)


def _validate_teacher_and_capacity(layout: ExperimentLayout) -> None:
    capacity = _load_json(layout.capacity / "gate.json")
    _require(capacity.get("status") == "passed", "capacity gate is not passed")
    warmstart = _load_json(layout.artifact_root / "warmstart_gate.json")
    _require(warmstart.get("status") == "passed", "warm-start gate is not passed")
    gate = _load_json(layout.teacher / "gate.json")
    route = _load_json(layout.teacher / "route.json")
    _require(gate.get("status") == "passed", "Teacher gate is not passed")
    _require(int(gate["development_correct"]) >= 52, "Teacher development threshold failed")
    _require(int(gate["actual_updates"]) == 100, "unexpected Teacher update count")
    _require(route.get("status") == "frozen", "Teacher route is not frozen")
    checkpoints = route.get("checkpoints")
    _require(isinstance(checkpoints, list), "Teacher route checkpoints are malformed")
    _require([int(item["step"]) for item in checkpoints] == [0, 20, 40, 60, 80, 100], "wrong Teacher route")
    for item in checkpoints:
        checkpoint = Path(item["checkpoint"])
        _require(_checkpoint_complete(checkpoint), f"incomplete Teacher checkpoint: {checkpoint}")
    _require(
        Path(gate["checkpoint"]).resolve() == Path(checkpoints[-1]["checkpoint"]).resolve(),
        "Teacher endpoint differs from route endpoint",
    )


def _curve_rows(layout: ExperimentLayout) -> dict[str, list[dict[str, Any]]]:
    path = layout.artifact_root / "student_curves.csv"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["condition"]].append(
                {
                    "step": int(row["step"]),
                    "correct": int(row["correct"]),
                    "total": int(row["total"]),
                    "accuracy": float(row["accuracy"]),
                }
            )
    return grouped


def _validate_curve_shape(
    layout: ExperimentLayout, expected_steps: dict[str, list[int]]
) -> dict[str, list[dict[str, Any]]]:
    grouped = _curve_rows(layout)
    known = set(METHOD_LABELS.values())
    _require(set(grouped) == known, f"unexpected curve conditions: {sorted(grouped)}")
    for label, steps in expected_steps.items():
        rows = grouped[label]
        observed = [row["step"] for row in rows]
        _require(len(observed) == len(set(observed)), f"duplicate curve milestone for {label}")
        _require(sorted(observed) == steps, f"wrong curve milestones for {label}: {observed}")
        for row in rows:
            _require(row["total"] == 64, f"wrong development total for {label}")
            _require(row["correct"] / row["total"] == row["accuracy"], f"wrong accuracy for {label}")
    return grouped


def _validate_selected(
    layout: ExperimentLayout,
    method: str,
    grouped: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    label = METHOD_LABELS[method]
    selected = _load_json(layout.method(method) / "selected.json")
    _require(selected.get("condition") == label, f"wrong selected condition for {method}")
    best = max(grouped[label], key=lambda row: (row["accuracy"], -row["step"]))
    _require(int(selected["selected_step"]) == best["step"], f"wrong selected step for {method}")
    _require(
        abs(float(selected["development_accuracy"]) - best["accuracy"]) < 1e-12,
        f"wrong selected accuracy for {method}",
    )
    expected = (
        layout.student_base
        if best["step"] == 0
        else layout.method(method) / f"step{best['step']}"
    )
    _require(
        Path(selected["checkpoint"]).resolve() == expected.resolve(),
        f"wrong selected checkpoint for {method}",
    )
    _require(_checkpoint_complete(expected), f"incomplete selected checkpoint for {method}")
    return selected


def _raw_rollout_stats(path: Path) -> dict[str, Any]:
    counts: Counter[int] = Counter()
    tokens: Counter[int] = Counter()
    lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            update = int(record["update"])
            completion = record["completion_ids"]
            _require(isinstance(completion, list), f"bad completion at {path}:{line_number}")
            counts[update] += 1
            tokens[update] += len(completion)
            lines += 1
    return {
        "sha256": _sha256(path),
        "lines": lines,
        "minimum_update": min(counts) if counts else None,
        "maximum_update": max(counts) if counts else None,
        "counts": {str(key): value for key, value in sorted(counts.items())},
        "tokens": {str(key): value for key, value in sorted(tokens.items())},
    }


def _all_raw_rollout_tokens(output: Path) -> int:
    paths = sorted(output.glob("raw_rollouts*.jsonl"))
    _require(bool(paths), f"no rollout logs found in {output}")
    total = 0
    for path in paths:
        stats = _raw_rollout_stats(path)
        total += sum(int(value) for value in stats["tokens"].values())
    return total
def _validate_contiguous_rollouts(
    stats: dict[str, Any], first: int, last: int
) -> None:
    expected = {str(update): 32 for update in range(first, last + 1)}
    _require(
        stats["counts"] == expected,
        f"rollout slots are not exactly 32 for updates {first}-{last}",
    )
    _require(stats["lines"] == (last - first + 1) * 32, "wrong rollout line count")


def _validate_selected_hashes(
    layout: ExperimentLayout, endpoint_sha: str, progressive_sha: str
) -> None:
    _require(
        len(endpoint_sha) == 64 and len(progressive_sha) == 64,
        "missing selected SHA",
    )
    _require(
        _sha256(layout.method("endpoint") / "selected.json") == endpoint_sha,
        "Endpoint selected.json changed",
    )
    _require(
        _sha256(layout.method("progressive") / "selected.json") == progressive_sha,
        "Progressive selected.json changed",
    )


def _pre_resume_initial(
    layout: ExperimentLayout,
    endpoint_sha: str,
    progressive_sha: str,
    source_job_id: str,
    prior_gpu_hours: float,
) -> None:
    _validate_teacher_and_capacity(layout)
    _validate_selected_hashes(layout, endpoint_sha, progressive_sha)
    _require(not layout.frozen_selection.exists(), "selection was already frozen")
    _require(not any(layout.final.glob("*.json")), "official-test output already exists")
    _require(
        not (layout.method("direct") / "selected.json").exists(),
        "Direct is already selected",
    )
    _require(
        not (layout.method("cafd") / "selected.json").exists(),
        "CAFD is already selected",
    )

    grouped = _validate_curve_shape(
        layout,
        {
            "Direct RLVR": [0, 40, 80],
            "Endpoint OPD": MILESTONES,
            "Progressive GKD": MILESTONES,
            "CAFD-v1": [0, 40, 80],
        },
    )
    _validate_selected(layout, "endpoint", grouped)
    _validate_selected(layout, "progressive", grouped)

    direct_resume = layout.method("direct") / "resume.pt"
    cafd_resume = layout.method("cafd") / "resume.pt"
    _require(
        direct_resume.stat().st_size > 20_000_000_000,
        "Direct resume is incomplete",
    )
    _require(cafd_resume.stat().st_size > 20_000_000_000, "CAFD resume is incomplete")
    _require(
        _checkpoint_complete(layout.method("direct") / "step120"),
        "Direct step120 is incomplete",
    )
    _require(
        _checkpoint_complete(layout.method("cafd") / "step80"),
        "CAFD step80 is incomplete",
    )
    phase_reference = (
        layout.method("cafd")
        / "phase_references"
        / "phase-02-reference.pt"
    )
    _require(
        phase_reference.stat().st_size > 3_000_000_000,
        "CAFD phase-02 reference is incomplete",
    )
    direct_state = _load_json(layout.state_root / "direct.json")
    cafd_state = _load_json(layout.state_root / "cafd.json")
    _require(
        int(direct_state["update"]) == 120,
        "unexpected Direct interrupted state",
    )
    _require(int(cafd_state["update"]) == 116, "unexpected CAFD interrupted state")

    direct_raw = _raw_rollout_stats(
        layout.method("direct") / "raw_rollouts.jsonl"
    )
    cafd_raw = _raw_rollout_stats(layout.method("cafd") / "raw_rollouts.jsonl")
    _validate_contiguous_rollouts(direct_raw, 1, 120)
    _validate_contiguous_rollouts(cafd_raw, 1, 116)
    cafd_tokens = {
        int(key): value for key, value in cafd_raw["tokens"].items()
    }
    discarded_generated = sum(
        value for update, value in cafd_tokens.items() if update > 80
    )
    recovery = {
        "status": "resume_authorized",
        "source_job_id": source_job_id,
        "source_job_state": "CANCELLED",
        "resume_job_id": os.environ.get("SLURM_JOB_ID"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "durable_resume_updates": {"direct": 120, "cafd": 80},
        "interrupted_observed_updates": {"direct": 120, "cafd": 116},
        "prior_gpu_hours_per_method": prior_gpu_hours,
        "old_raw_rollouts": {"direct": direct_raw, "cafd": cafd_raw},
        "discarded_cafd_tail": {
            "updates": [81, 116],
            "generated_tokens": discarded_generated,
            "teacher_scored_tokens": 2 * discarded_generated,
        },
        "endpoint_selected_sha256": endpoint_sha,
        "progressive_selected_sha256": progressive_sha,
    }
    write_json(layout.run_root / "recovery.json", recovery)


def _pre_resume(
    layout: ExperimentLayout,
    endpoint_sha: str,
    progressive_sha: str,
    source_job_id: str,
    prior_gpu_hours: float,
) -> None:
    recovery_path = layout.run_root / "recovery.json"
    if not recovery_path.exists():
        _pre_resume_initial(
            layout,
            endpoint_sha,
            progressive_sha,
            source_job_id,
            prior_gpu_hours,
        )

    _validate_teacher_and_capacity(layout)
    _validate_selected_hashes(layout, endpoint_sha, progressive_sha)
    _require(not layout.frozen_selection.exists(), "selection was already frozen")
    _require(
        not any(layout.final.glob("*.json")),
        "official-test output already exists",
    )
    recovery = _load_json(recovery_path)
    _require(
        recovery["source_job_id"] == source_job_id,
        "recovery source job changed",
    )
    _require(
        recovery["endpoint_selected_sha256"] == endpoint_sha,
        "recovery Endpoint SHA changed",
    )
    _require(
        recovery["progressive_selected_sha256"] == progressive_sha,
        "recovery Progressive SHA changed",
    )
    for method in ("direct", "cafd"):
        current = _raw_rollout_stats(
            layout.method(method) / "raw_rollouts.jsonl"
        )
        baseline = recovery["old_raw_rollouts"][method]
        _require(
            current["sha256"] == baseline["sha256"]
            and current["lines"] == baseline["lines"],
            f"{method} baseline rollout log changed",
        )

    observed_rows = _curve_rows(layout)
    expected_steps = {
        label: sorted(row["step"] for row in observed_rows[label])
        for label in METHOD_LABELS.values()
    }
    grouped = _validate_curve_shape(layout, expected_steps)
    for method in ("endpoint", "progressive"):
        label = METHOD_LABELS[method]
        _require(
            expected_steps[label] == MILESTONES,
            f"{method} curve is no longer complete",
        )
        _validate_selected(layout, method, grouped)

    resume_updates: dict[str, int] = {}
    baseline_updates = recovery["durable_resume_updates"]
    for method in ("direct", "cafd"):
        label = METHOD_LABELS[method]
        observed = expected_steps[label]
        _require(
            len(observed) >= 3
            and observed == MILESTONES[: len(observed)],
            f"{method} curve is not a milestone prefix",
        )
        selected_path = layout.method(method) / "selected.json"
        if selected_path.exists():
            _require(
                observed == MILESTONES,
                f"{method} is selected with an incomplete curve",
            )
            _validate_selected(layout, method, grouped)
            resume_updates[method] = 200
            continue
        resume_path = layout.method(method) / "resume.pt"
        _require(
            resume_path.stat().st_size > 20_000_000_000,
            f"{method} resume is incomplete",
        )
        resume_update = _resume_update(resume_path, method)
        _require(
            resume_update in MILESTONES
            and resume_update >= int(baseline_updates[method]),
            f"{method} durable resume update is invalid: {resume_update}",
        )
        _require(
            observed[-1] <= resume_update,
            f"{method} curve is ahead of its durable resume",
        )
        for step in observed[1:]:
            _require(
                _checkpoint_complete(layout.method(method) / f"step{step}"),
                f"{method} checkpoint step{step} is incomplete",
            )
        resume_updates[method] = resume_update

    job_id = os.environ.get("SLURM_JOB_ID")
    attempts = recovery.setdefault("attempts", [])
    if job_id and not any(item.get("job_id") == job_id for item in attempts):
        attempts.append(
            {
                "job_id": job_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "resume_updates": resume_updates,
                "prior_gpu_hours_per_method": prior_gpu_hours,
            }
        )
    recovery["status"] = "resume_authorized"
    recovery["latest_resume_updates"] = resume_updates
    recovery["latest_resume_job_id"] = job_id
    write_json(recovery_path, recovery)


def _post_students(
    layout: ExperimentLayout, endpoint_sha: str, progressive_sha: str
) -> None:
    _validate_teacher_and_capacity(layout)
    _validate_selected_hashes(layout, endpoint_sha, progressive_sha)
    grouped = _validate_curve_shape(
        layout, {label: MILESTONES for label in METHOD_LABELS.values()}
    )
    selected = {
        method: _validate_selected(layout, method, grouped)
        for method in METHOD_LABELS
    }
    recovery = _load_json(layout.run_root / "recovery.json")
    baseline_gpu_hours = float(recovery["prior_gpu_hours_per_method"])
    baseline_updates = recovery["durable_resume_updates"]
    baseline_discarded = {
        "direct": {"generated_tokens": 0, "teacher_scored_tokens": 0},
        "cafd": recovery["discarded_cafd_tail"],
    }
    for method in ("direct", "cafd"):
        record = selected[method]
        start_update = int(record.get("resume_start_update", -1))
        _require(
            start_update in MILESTONES
            and start_update >= int(baseline_updates[method]),
            f"wrong resume start for {method}: {start_update}",
        )
        _require(
            float(record.get("prior_gpu_hours", -1.0)) >= baseline_gpu_hours,
            f"prior GPU-hours dropped for {method}",
        )
        _require(
            float(record["gpu_hours"]) >= float(record["prior_gpu_hours"]),
            f"total GPU-hours dropped for {method}",
        )
        logical_generated = int(record["logical_generated_tokens"])
        logical_scored = int(record["logical_teacher_scored_tokens"])
        discarded_generated = int(record["prior_discarded_generated_tokens"])
        discarded_scored = int(record["prior_discarded_teacher_scored_tokens"])
        _require(
            discarded_generated
            >= int(baseline_discarded[method]["generated_tokens"]),
            f"discarded generated tokens dropped for {method}",
        )
        _require(
            discarded_scored
            >= int(baseline_discarded[method]["teacher_scored_tokens"]),
            f"discarded teacher-scored tokens dropped for {method}",
        )
        _require(
            int(record["actual_generated_tokens"])
            == logical_generated + discarded_generated,
            f"wrong actual generated tokens for {method}",
        )
        _require(
            int(record["actual_teacher_scored_tokens"])
            == logical_scored + discarded_scored,
            f"wrong actual teacher-scored tokens for {method}",
        )
        physical_generated = _all_raw_rollout_tokens(layout.method(method))
        scored_multiplier = {"direct": 0, "cafd": 2}[method]
        _require(
            int(record["actual_generated_tokens"]) == physical_generated,
            f"actual generated tokens do not match rollout logs for {method}",
        )
        _require(
            int(record["actual_teacher_scored_tokens"])
            == physical_generated * scored_multiplier,
            f"actual teacher-scored tokens do not match rollout logs for {method}",
        )
        if start_update < 200:
            attempt = Path(record.get("rollout_log", ""))
            _require(
                attempt.is_file() and attempt.name != "raw_rollouts.jsonl",
                f"missing isolated recovery rollout log for {method}",
            )
            stats = _raw_rollout_stats(attempt)
            _validate_contiguous_rollouts(stats, start_update + 1, 200)

    for method in ("direct", "cafd"):
        old_path = layout.method(method) / "raw_rollouts.jsonl"
        current = _raw_rollout_stats(old_path)
        original = recovery["old_raw_rollouts"][method]
        _require(
            current["sha256"] == original["sha256"],
            f"{method} old rollout hash changed",
        )
        _require(
            current["lines"] == original["lines"],
            f"{method} old rollout line count changed",
        )


def _post_freeze(
    layout: ExperimentLayout, endpoint_sha: str, progressive_sha: str
) -> None:
    _post_students(layout, endpoint_sha, progressive_sha)
    frozen = _load_json(layout.frozen_selection)
    frozen_conditions = frozen.get("conditions")
    _require(isinstance(frozen_conditions, dict), "missing frozen conditions")
    _require(
        set(frozen_conditions) == set(FINAL_CONDITIONS),
        "wrong frozen condition set",
    )
    for method in METHOD_LABELS:
        selected = _load_json(layout.method(method) / "selected.json")
        record = frozen["conditions"][method]
        _require(int(record["budget"]) == 200, f"wrong frozen budget for {method}")
        _require(
            int(record["selected_step"]) == int(selected["selected_step"]),
            f"wrong frozen step for {method}",
        )
        _require(
            Path(record["checkpoint"]).resolve()
            == Path(selected["checkpoint"]).resolve(),
            f"wrong frozen checkpoint for {method}",
        )
        for key in (
            "generated_tokens",
            "teacher_scored_tokens",
            "logical_generated_tokens",
            "logical_teacher_scored_tokens",
            "actual_generated_tokens",
            "actual_teacher_scored_tokens",
            "gpu_hours",
        ):
            if key in selected:
                _require(
                    record.get(key) == selected[key],
                    f"frozen {key} differs from selection for {method}",
                )


def _post_final(
    layout: ExperimentLayout, endpoint_sha: str, progressive_sha: str
) -> None:
    _post_freeze(layout, endpoint_sha, progressive_sha)
    frozen = _load_json(layout.frozen_selection)["conditions"]
    final_records: dict[str, dict[str, Any]] = {}
    for condition in FINAL_CONDITIONS:
        record = _load_json(layout.final / f"{condition}.json")
        final_records[condition] = record
        _require(
            record["condition"] == condition,
            f"wrong final condition: {condition}",
        )
        _require(
            int(record["total"]) == 132,
            f"wrong official test total: {condition}",
        )
        for key in ("label", "checkpoint", "selected_step", "budget"):
            _require(
                record[key] == frozen[condition][key],
                f"final {key} differs from frozen selection: {condition}",
            )
        for key in (
            "generated_tokens",
            "teacher_scored_tokens",
            "logical_generated_tokens",
            "logical_teacher_scored_tokens",
            "actual_generated_tokens",
            "actual_teacher_scored_tokens",
            "gpu_hours",
        ):
            if key in frozen[condition]:
                _require(
                    record.get(key) == frozen[condition][key],
                    f"final {key} differs from frozen selection: {condition}",
                )
        _require(
            int(record["evaluation_generated_tokens"]) >= 0,
            f"missing held-out inference token accounting: {condition}",
        )
        _require(
            int(record["correct"]) / int(record["total"])
            == float(record["accuracy"]),
            f"wrong official accuracy: {condition}",
        )
    for name in ("final_scores.csv", "costs.csv", "report.md", "result.png"):
        _require(
            (layout.artifact_root / name).is_file(),
            f"missing final artifact: {name}",
        )
    with (layout.artifact_root / "final_scores.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        scores = list(csv.DictReader(handle))
    with (layout.artifact_root / "costs.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        costs = list(csv.DictReader(handle))
    _require(
        len(scores) == len(FINAL_CONDITIONS),
        "wrong final score row count",
    )
    _require(
        len(costs) == len(FINAL_CONDITIONS),
        "wrong final cost row count",
    )

    score_map = {row["condition"]: row for row in scores}
    cost_map = {row["condition"]: row for row in costs}
    for condition, record in final_records.items():
        label = record["label"]
        _require(label in score_map and label in cost_map, f"missing CSV row: {label}")
        score = score_map[label]
        _require(
            int(score["correct"]) == int(record["correct"])
            and int(score["total"]) == int(record["total"])
            and float(score["test_full_pass"]) == float(record["accuracy"])
            and int(score["selected_step"]) == int(record["selected_step"])
            and int(score["budget_updates"]) == int(record["budget"])
            and Path(score["checkpoint"]).resolve()
            == Path(record["checkpoint"]).resolve(),
            f"final score CSV mismatch: {label}",
        )
        logical_generated = int(
            record.get("logical_generated_tokens", record.get("generated_tokens", 0))
        )
        logical_scored = int(
            record.get(
                "logical_teacher_scored_tokens",
                record.get("teacher_scored_tokens", 0),
            )
        )
        actual_generated = int(
            record.get("actual_generated_tokens", logical_generated)
        )
        actual_scored = int(
            record.get("actual_teacher_scored_tokens", logical_scored)
        )
        cost = cost_map[label]
        _require(
            int(cost["generated_tokens"]) == logical_generated
            and int(cost["teacher_scored_tokens"]) == logical_scored
            and int(cost["logical_generated_tokens"]) == logical_generated
            and int(cost["logical_teacher_scored_tokens"]) == logical_scored
            and int(cost["actual_generated_tokens"]) == actual_generated
            and int(cost["actual_teacher_scored_tokens"]) == actual_scored
            and abs(float(cost["gpu_hours"]) - float(record.get("gpu_hours", 0.0)))
            < 1e-9,
            f"final cost CSV mismatch: {label}",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("pre-resume", "post-students", "post-freeze", "post-final"),
        required=True,
    )
    parser.add_argument("--endpoint-sha", required=True)
    parser.add_argument("--progressive-sha", required=True)
    parser.add_argument("--source-job-id", default="40852412")
    parser.add_argument(
        "--prior-gpu-hours",
        type=float,
        default=5.639166666666667,
    )
    args = parser.parse_args()
    _require(args.prior_gpu_hours >= 0.0, "invalid prior GPU-hours")

    root = args.root.resolve()
    _, layout = _protocol(root)
    if args.phase == "pre-resume":
        _pre_resume(
            layout,
            args.endpoint_sha,
            args.progressive_sha,
            args.source_job_id,
            args.prior_gpu_hours,
        )
    elif args.phase == "post-students":
        _post_students(
            layout,
            args.endpoint_sha,
            args.progressive_sha,
        )
    elif args.phase == "post-freeze":
        _post_freeze(
            layout,
            args.endpoint_sha,
            args.progressive_sha,
        )
    else:
        _post_final(
            layout,
            args.endpoint_sha,
            args.progressive_sha,
        )
    print(
        json.dumps(
            {"phase": args.phase, "status": "passed"},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
