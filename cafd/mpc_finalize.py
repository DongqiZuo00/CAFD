"""One development-frozen CAFD-MPC endpoint evaluation; never select on test.

The default CLI safely continues a partially written per-task journal. Admission
checks use only frozen metadata; test labels and CUDA models are accessed only
after a complete 200-round formal run and matching frozen selection are verified.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .mpc_data import checkpoint_metadata

ROOT = Path("/blue/du.j/jinjiaguo/CAFD")
S0_REL = Path("runs/cafd/experiments/mistral_cafd_disjoint_v7/student_base/S0")
TEST_REL = Path("data/cafd/test.jsonl")
IDS_REL = Path("data/cafd/mpc_v1/test_ids.json")
FAMILIES = ("contains_count", "contains_ordered", "contains_substring")


def _inside(root: Path, value: str | Path) -> Path:
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise RuntimeError(f"path escapes CAFD root: {value}")
    return path


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _file_metadata(root: Path, path: Path) -> dict[str, Any]:
    stat = path.stat()
    if not path.is_file() or stat.st_size <= 0:
        raise RuntimeError(f"missing/empty frozen source: {path}")
    return {"path": str(path.relative_to(root)), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def validate_finalization(root: Path, run_id: str) -> dict[str, Any]:
    """Read identities/config/index/IDs only, NOT test question or label payload."""
    root = root.resolve()
    if not re.fullmatch(r"mistral_cafd_mpc_[a-zA-Z0-9_]+", run_id) or run_id.endswith("_smoke"):
        raise RuntimeError("formal MPC run identity required")
    run = _inside(root, Path("runs/cafd/experiments") / run_id)
    artifact = _inside(root, Path("artifacts/cafd/experiments") / run_id)
    complete = _read(run / "complete.json")
    selected = _read(run / "selected.json")
    cfg = _read(run / "config.json")
    manifest = _read(run / "data_manifest.json")
    if (complete.get("status") != "STUDENT_TRAINING_COMPLETE" or
            complete.get("formal_result") is not True or type(complete.get("rounds")) is not int or
            complete["rounds"] != 200):
        raise RuntimeError("test requires completed formal 200-round Student training")
    actual = complete.get("actual_updates")
    if type(actual) is not int or not 0 <= actual <= 200:
        raise RuntimeError("invalid actual optimizer-update budget")
    if (cfg.get("profile") != "formal" or cfg.get("protocol") != "CAFD-MPC-2026-09-06" or
            cfg.get("total_rounds") != 200 or cfg.get("max_new_tokens") != 2048 or
            cfg.get("seed") != 2027 or cfg.get("max_prompt_tokens") != 4096 or
            cfg.get("prompts_per_round") != 4 or cfg.get("rollouts_per_prompt") != 8 or
            cfg.get("block_rounds") != 10 or cfg.get("horizon") != 2 or
            cfg.get("milestones") != [0, 10, 40, 80, 120, 160, 200]):
        raise RuntimeError("formal evaluation configuration changed")
    if selected.get("status") != "frozen" or selected.get("selection_rule") != "development_full_pass_earliest_tie":
        raise RuntimeError("development selection is not frozen")
    best = {key: value for key, value in selected.items() if key not in {"status", "selection_rule"}}
    if best != complete.get("selected"):
        raise RuntimeError("selected.json differs from complete.selected")
    step = best.get("round")
    if type(step) is not int or step not in cfg["milestones"]:
        raise RuntimeError("selected checkpoint is not a declared development milestone")
    if (best.get("total") != 64 or type(best.get("correct")) is not int or
            not 0 <= best["correct"] <= 64 or
            best.get("accuracy") != best["correct"] / 64):
        raise RuntimeError("invalid frozen development-selection score")
    checkpoint = _inside(root, best.get("checkpoint", ""))
    if step == 0:
        if checkpoint != _inside(root, S0_REL):
            raise RuntimeError("round zero can select only unchanged original S0")
    elif checkpoint != _inside(root, run / f"round{step}"):
        raise RuntimeError("selected checkpoint does not belong to this formal run/round")
    metadata = checkpoint_metadata(root, checkpoint)
    if manifest.get("version") != "cafd_mpc_v1" or manifest.get("status") != "frozen":
        raise RuntimeError("MPC data manifest is not frozen")
    test = manifest.get("official_test", {})
    if test.get("count") != 132 or test.get("ids_metadata_path") != str(IDS_REL):
        raise RuntimeError("wrong frozen official-test identity")
    source = _inside(root, TEST_REL)
    if test.get("source_file_metadata") != _file_metadata(root, source):
        raise RuntimeError("official-test source metadata changed after data freeze")
    ids_meta = _read(_inside(root, IDS_REL))
    ids = ids_meta.get("ids")
    if (ids_meta.get("source") != str(TEST_REL) or ids_meta.get("count") != 132 or
            not isinstance(ids, list) or len(ids) != 132 or
            any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != 132):
        raise RuntimeError("invalid frozen ID-only test metadata")
    for split in ("optimization", "control", "development"):
        if set(manifest["partitions"][split]["ids"]) & set(ids):
            raise RuntimeError(f"test IDs overlap {split}")
    identity = {
        "version": "cafd_mpc_final_v1", "run_id": run_id, "protocol": cfg["protocol"],
        "checkpoint": str(checkpoint), "checkpoint_metadata": metadata,
        "selection": selected, "training_rounds": 200, "actual_optimizer_updates": actual,
        "seed": 2027, "generation": {"max_new_tokens": 2048, "max_prompt_tokens": 4096,
            "sample": False, "num_return_sequences": 1,
            "renderer": "ministral3_instruct_fixed_system",
            "stop": cfg.get("output_stop"), "verifier": cfg.get("reward")},
        "test_ids": ids, "test_source_metadata": test["source_file_metadata"],
        "test_historically_evaluated": bool(test.get("historically_evaluated", True)),
    }
    return {"root": str(root), "run": str(run), "artifact": str(artifact),
            "test_source": str(source), "identity": identity,
            "training_costs": complete.get("costs", {}), "training_job_id": complete.get("job_id"),
            "limitations": manifest.get("limitations", []), "config": cfg}


def load_test_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Called only after validate_finalization: labels used for scoring, never selection."""
    with Path(context["test_source"]).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if [row["id"] for row in rows] != context["identity"]["test_ids"]:
        raise RuntimeError("official test payload ID order differs from frozen ID-only manifest")
    if any(row.get("problem_family") not in FAMILIES for row in rows):
        raise RuntimeError("unexpected official-test problem family")
    return rows


def _publish_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"refusing to overwrite frozen final artifact: {path}")
        return
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_record(record: dict[str, Any], index: int, context: dict[str, Any]) -> None:
    identity = context["identity"]
    if (record.get("index") != index or record.get("row_id") != identity["test_ids"][index] or
            record.get("checkpoint") != identity["checkpoint"] or record.get("run_id") != identity["run_id"]):
        raise RuntimeError("final output journal identity/order mismatch")
    if record.get("family") not in FAMILIES:
        raise RuntimeError("unknown output problem family")
    reward = record.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward) or not 0 <= reward <= 1:
        raise RuntimeError("invalid output reward")
    if type(record.get("full_pass")) is not bool or record["full_pass"] != (reward == 1):
        raise RuntimeError("verifier success and full reward disagree")
    for key in ("prompt_ids", "completion_ids"):
        tokens = record.get(key)
        if not isinstance(tokens, list) or any(type(token) is not int or not 0 <= token < 131072 for token in tokens):
            raise RuntimeError(f"invalid persisted {key}")
    if len(record["completion_ids"]) > 2048:
        raise RuntimeError("persisted output exceeds frozen generation budget")
    if record.get("failure_category") != failure_category(record):
        raise RuntimeError("persisted failure category differs from verifier flags")


def read_outputs(path: Path, context: dict[str, Any], *, repair_tail: bool = True) -> list[dict[str, Any]]:
    """Recover only a syntactically incomplete LAST line; preserve its original bytes."""
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("output journal must be a regular non-symlink file")
    records = []
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                remainder = handle.read()
                if remainder or not repair_tail:
                    raise RuntimeError("corrupt interior final output journal")
                archive = path.parent / "final_test_recovery"
                archive.mkdir(exist_ok=True)
                _publish_once(archive / f"tail-{time.time_ns()}.hex", raw.hex() + "\n")
                with path.open("r+b") as target:
                    target.truncate(offset)
                    target.flush()
                    os.fsync(target.fileno())
                break
            if len(records) >= 132 or not isinstance(record, dict):
                raise RuntimeError("too many or malformed final output records")
            _validate_record(record, len(records), context)
            records.append(record)
            if not raw.endswith(b"\n"):
                # A complete final JSON line survived without its newline.
                with path.open("ab") as target:
                    target.write(b"\n")
                    target.flush()
                    os.fsync(target.fileno())
                break
    return records


def failure_category(record: dict[str, Any]) -> str:
    if record.get("full_pass"):
        return "passed"
    if record.get("length_cap"):
        return "truncation"
    if record.get("tier") == "invalid_format":
        return "format"
    if record.get("tier") == "format_only":
        return "parse"
    return "semantic"


def _add(costs, key, value):
    if hasattr(costs, "add"):
        costs.add(key, value)
    else:
        costs[key] = costs.get(key, 0) + value


def evaluate_remaining(context, rows, model, tokenizer, costs, *, generate_fn=None):
    if generate_fn is None:
        from .mpc_runtime import generate as generate_fn
    artifact = Path(context["artifact"])
    identity = context["identity"]
    _publish_once(artifact / "finalization_identity.json", _pretty(identity))
    path = artifact / "final_test_outputs.jsonl"
    records = read_outputs(path, context)
    if [row["id"] for row in rows] != identity["test_ids"]:
        raise RuntimeError("evaluation rows do not match frozen test IDs")
    for old, row in zip(records, rows):
        if old["family"] != row["problem_family"]:
            raise RuntimeError("persisted family differs from official task")
    for index in range(len(records), len(rows)):
        started = time.monotonic()
        generated = generate_fn(model, tokenizer, rows[index], count=1, max_new_tokens=2048,
            max_prompt_tokens=4096, seed=2027 + 3_000_000 + index, sample=False)
        if len(generated) != 1:
            raise RuntimeError("official pass@1 requires exactly one completion")
        record = dict(generated[0], index=index, checkpoint=identity["checkpoint"],
                      run_id=identity["run_id"], evaluation_seconds=time.monotonic() - started)
        record["failure_category"] = failure_category(record)
        _validate_record(record, index, context)
        # Physical work is recorded before publishing a task result. A failed
        # publish/retry may cost extra; those counters must NOT roll back.
        _add(costs, "evaluation_generated_tokens", len(record["completion_ids"]))
        _add(costs, "evaluation_prompt_tokens", len(record["prompt_ids"]))
        _add(costs, "evaluation_verifier_calls", 1)
        _add(costs, "evaluation_operation_seconds", record["evaluation_seconds"])
        _append_record(path, record)
        records.append(record)
        from .training_common import write_json
        write_json(artifact / "final_test_progress.json", {
            "status": "evaluating", "completed": len(records), "total": 132,
            "checkpoint": identity["checkpoint"], "run_id": identity["run_id"],
            "selection_changed": False,
        })
    return records


def aggregate(context, records, costs) -> dict[str, Any]:
    if len(records) != 132:
        raise RuntimeError("cannot finalize an incomplete official test")
    for index, record in enumerate(records):
        _validate_record(record, index, context)
    correct = sum(record["full_pass"] for record in records)
    families = []
    for family in FAMILIES:
        subset = [record for record in records if record["family"] == family]
        count = len(subset)
        passed = sum(record["full_pass"] for record in subset)
        families.append({"family": family, "correct": passed, "total": count,
                         "accuracy": passed / count if count else None,
                         "failure_counts": dict(Counter(record["failure_category"] for record in subset))})
    identity = context["identity"]
    train = context["training_costs"]
    row = {
        "condition": "CAFD-MPC", "benchmark": "DELTA Manufactoria-HAS",
        "backbone": "mistralai/Ministral-3-3B-Instruct-2512-BF16",
        "seed": 2027, "correct": correct, "total": 132, "accuracy": correct / 132,
        "training_rounds": 200, "actual_optimizer_updates": identity["actual_optimizer_updates"],
        "selected_round": identity["selection"]["round"], "checkpoint": identity["checkpoint"],
        "training_generated_tokens": train.get("training_generated_tokens"),
        "training_teacher_scored_tokens": train.get("training_teacher_scored_tokens", 0),
        "training_gpu_hours": train.get("gpu_hours"),
        "evaluation_generated_tokens": costs.get("evaluation_generated_tokens", sum(len(r["completion_ids"]) for r in records)),
        "evaluation_gpu_hours_lower_bound": (
            costs.get("evaluation_operation_seconds", 0.) + costs.get("evaluation_model_loading_seconds", 0.)) / 3600,
        "combined_allocated_gpu_hours": costs.get("combined_allocated_gpu_hours"),
    }
    return {
        "status": "FINAL_TEST_COMPLETE", "identity": identity, "scores": row,
        "by_family": families, "failure_counts": dict(Counter(r["failure_category"] for r in records)),
        "evaluation_costs": dict(costs), "training_costs": train,
        "metric": "official full-pass/pass@1 under frozen response contract",
        "sampling": "greedy, one completion per task, cap 2048",
        "frozen_test_evaluated": True, "checkpoint_selection_changed": False,
        "historically_untouched_test": False,
        "limitations": context["limitations"],
    }


def _csv(rows: list[dict[str, Any]], fields: list[str]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in fields})
    return handle.getvalue()


def seal_result(context, records, costs) -> dict[str, Any]:
    artifact = Path(context["artifact"])
    frozen = artifact / "final_test_summary.json"
    if frozen.exists():
        summary = _read(frozen)
        if summary.get("identity") != context["identity"]:
            raise RuntimeError("frozen final summary belongs to another selection")
        reconstructed = aggregate(context, records, summary["evaluation_costs"])
        if reconstructed != summary:
            raise RuntimeError("frozen final summary no longer matches task outputs")
    else:
        summary = aggregate(context, records, costs)
        _publish_once(frozen, _pretty(summary))
    row = summary["scores"]
    _publish_once(artifact / "final_scores.csv", _csv([row], list(row)))
    _publish_once(artifact / "final_scores_by_family.csv",
                  _csv(summary["by_family"], ["family", "correct", "total", "accuracy"]))
    _publish_once(artifact / "final_test_costs.json", _pretty({
        "evaluation": summary["evaluation_costs"], "training": summary["training_costs"]}))
    # This final JSON is the completion marker, published only after all sidecars.
    _publish_once(artifact / "final_scores.json", _pretty(summary))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="mistral_cafd_mpc_v1_formal")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("all work must remain in /blue/du.j/jinjiaguo/CAFD")
    context = validate_finalization(ROOT, args.run_id)
    artifact = Path(context["artifact"])
    artifact.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (artifact / "final_test.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _publish_once(artifact / "finalization_identity.json", _pretty(context["identity"]))
        records = read_outputs(artifact / "final_test_outputs.jsonl", context)
        if len(records) == 132:
            # Idempotent completion/recovery needs neither CUDA nor test labels.
            snapshot = _read(artifact / "final_test_summary.json") if (artifact / "final_test_summary.json").exists() else None
            if snapshot is None:
                from .mpc_ledger import CostLedger
                costs = CostLedger(artifact / "final_test_physical_costs.jsonl")
            else:
                costs = snapshot["evaluation_costs"]
            result = seal_result(context, records, costs)
            print(json.dumps(result["scores"], sort_keys=True), flush=True)
            return
        if (artifact / "final_scores.json").exists():
            raise RuntimeError("completed final result has an incomplete/corrupt task journal")
        import torch
        from .mpc_allocation import validate_allocation
        from .mpc_ledger import CostLedger, allocation_seconds
        from .mpc_runtime import load_hidden
        from .mistral_runtime import load_tokenizers, assert_exact_tokenizer_pair
        from .prompting import assert_matches_mistral_chat_template
        from .training_common import write_json
        if not torch.cuda.is_available():
            raise RuntimeError("new test generation requires an authorized Slurm B200 allocation")
        allocation = validate_allocation(os.environ, torch.cuda.get_device_name(0), torch.cuda.device_count())
        write_json(artifact / "final_test_allocation.json", allocation)
        costs = CostLedger(artifact / "final_test_physical_costs.jsonl")
        costs.begin_attempt(os.environ["SLURM_JOB_ID"], resume=bool(records) or bool(costs.attempts))
        rows = load_test_rows(context)
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        started = time.monotonic()
        teacher_tokenizer, tokenizer = load_tokenizers(ROOT / ".cache/huggingface/hub")
        model = load_hidden(ROOT, context["identity"]["checkpoint"], torch.device("cuda:0"), trainable=False)
        assert_exact_tokenizer_pair(teacher_tokenizer, tokenizer, student_model=model.causal_lm)
        assert_matches_mistral_chat_template(tokenizer)
        _add(costs, "evaluation_model_loading_seconds", time.monotonic() - started)
        records = evaluate_remaining(context, rows, model, tokenizer, costs)
        costs["evaluation_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        job_ids = costs.job_ids
        train_ledger = artifact / "physical_costs.jsonl"
        if train_ledger.exists():
            job_ids = list(dict.fromkeys(CostLedger(train_ledger).job_ids + job_ids))
        elif context.get("training_job_id"):
            job_ids = list(dict.fromkeys([str(context["training_job_id"])] + job_ids))
        allocated = allocation_seconds(job_ids)
        costs["combined_allocated_gpu_hours"] = None if allocated is None else allocated / 3600
        costs["combined_gpu_time_basis"] = "Slurm_allocated_elapsed" if allocated is not None else "unavailable"
        result = seal_result(context, records, costs)
        write_json(artifact / "final_test_progress.json", {
            "status": "complete", "completed": 132, "total": 132,
            "correct": result["scores"]["correct"], "checkpoint": context["identity"]["checkpoint"]})
        print(json.dumps(result["scores"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
