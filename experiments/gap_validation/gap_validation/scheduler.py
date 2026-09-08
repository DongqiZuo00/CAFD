from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from .io import ARTIFACT_ROOT, ROOT, atomic_json, load_yaml, read_json


MANIFEST = ARTIFACT_ROOT / "run_manifest.json"
CONTROL = ARTIFACT_ROOT / "scheduler_control.json"
ALLOWED_STATES = {
    "PENDING", "RUNNING", "RETRYING", "COMPLETED", "FAILED", "RESOURCE_BLOCKED",
    "DATA_CONTRACT_BLOCKED", "INVALID_TOKENIZER_PAIR", "BUDGET_INVALID",
    "DEFERRED_BY_SCOPE_REDUCTION",
}


def initialize() -> dict[str, Any]:
    jobs = load_yaml("jobs.yaml")["jobs"]
    existing = read_json(MANIFEST, {})
    manifest = {
        "schema_version": 1,
        "created_at": existing.get("created_at", time.time()),
        "updated_at": time.time(),
        "hard_gpu_ceiling": 4,
        "jobs": existing.get("jobs", {}),
    }
    for name, spec in jobs.items():
        current = manifest["jobs"].setdefault(name, {})
        current.setdefault("state", spec.get("state", "PENDING"))
        current.setdefault("dependencies", spec.get("dependencies", []))
        current.setdefault("gpus", int(spec.get("gpus", 0)))
        if spec.get("slurm_job_id"):
            configured_job_id = str(spec["slurm_job_id"])
            if current.get("slurm_job_id") != configured_job_id:
                current["slurm_job_id"] = configured_job_id
                current["state"] = spec.get("state", "RUNNING")
        if spec.get("blocker"):
            current.setdefault("blocker", spec["blocker"])
        if (
            spec.get("state") == "DEFERRED_BY_SCOPE_REDUCTION"
            and current.get("state") not in {"RUNNING", "RETRYING", "COMPLETED"}
        ):
            current["state"] = "DEFERRED_BY_SCOPE_REDUCTION"
            current["deferred_reason"] = spec.get("deferred_reason", "scope_reduction")
        current["spec"] = spec
    atomic_json(MANIFEST, manifest)
    return manifest


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def _requested_gpus() -> int:
    user = os.environ.get("USER", "jinjiaguo")
    result = _run(["squeue", "-h", "-u", user, "-p", "hpg-b200", "-o", "%b"])
    total = 0
    for line in result.stdout.splitlines():
        match = re.search(r"(?:gpu|gres/gpu)(?::b200)?:(\d+)", line)
        if match:
            total += int(match.group(1))
    return total


def refresh(manifest: dict[str, Any]) -> dict[str, Any]:
    for job in manifest["jobs"].values():
        job_id = job.get("slurm_job_id")
        if not job_id or job["state"] not in {"RUNNING", "RETRYING"}:
            continue
        result = _run(["sacct", "-n", "-X", "-j", str(job_id), "-o", "State,ExitCode", "-P"])
        first = next((line for line in result.stdout.splitlines() if line.strip()), "")
        state = first.split("|", 1)[0].split()[0] if first else ""
        if state == "COMPLETED":
            job["state"] = "COMPLETED"
        elif state in {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED", "NODE_FAIL"}:
            job["state"] = "FAILED"
            job["slurm_state"] = state
            job["exit_code"] = first.split("|", 1)[1] if "|" in first else None
    terminal_unsuccessful = {
        "FAILED", "RESOURCE_BLOCKED", "DATA_CONTRACT_BLOCKED", "INVALID_TOKENIZER_PAIR", "BUDGET_INVALID"
    }
    for job in manifest["jobs"].values():
        if job["state"] != "PENDING":
            continue
        blockers = [
            dependency
            for dependency in job["dependencies"]
            if manifest["jobs"].get(dependency, {}).get("state") in terminal_unsuccessful
        ]
        if blockers:
            job["state"] = "FAILED"
            job["blocker"] = "DEPENDENCY_NOT_COMPLETED"
            job["blocked_by"] = blockers
    manifest["updated_at"] = time.time()
    atomic_json(MANIFEST, manifest)
    return manifest


def dispatch() -> dict[str, Any]:
    manifest = refresh(initialize())
    control = read_json(CONTROL, {})
    if control.get("mode") == "DRAIN":
        manifest["scheduler_mode"] = "DRAIN"
        manifest["drain_reason"] = control.get("reason")
        manifest["updated_at"] = time.time()
        atomic_json(MANIFEST, manifest)
        return manifest
    active_gpus = _requested_gpus()
    for name, job in manifest["jobs"].items():
        if job["state"] != "PENDING" or "script" not in job["spec"]:
            continue
        if any(manifest["jobs"].get(dep, {}).get("state") != "COMPLETED" for dep in job["dependencies"]):
            continue
        gpus = int(job["spec"].get("gpus", 0))
        if active_gpus + gpus > 4:
            continue
        spec = job["spec"]
        command = [
            "sbatch", "--parsable", f"--partition={spec['partition']}", "--account=du.j", "--qos=du.j",
            f"--job-name=gap-{name}", f"--cpus-per-task={spec['cpus']}", f"--mem={spec['memory_gb']}gb",
            f"--time={spec['time']}", f"--output={ROOT}/logs/gap_validation/{name}-%j.out",
            f"--error={ROOT}/logs/gap_validation/{name}-%j.err",
        ]
        if gpus:
            command.append(f"--gpus={gpus}")
        command.append(str(ROOT / spec["script"]))
        result = _run(command)
        if result.returncode != 0:
            job["last_submit_error"] = result.stderr.strip()
            continue
        job["slurm_job_id"] = result.stdout.strip().split(";", 1)[0]
        job["state"] = "RUNNING"
        job["submitted_at"] = time.time()
        job["stdout"] = f"logs/gap_validation/{name}-{job['slurm_job_id']}.out"
        job["stderr"] = f"logs/gap_validation/{name}-{job['slurm_job_id']}.err"
        active_gpus += gpus
    manifest["updated_at"] = time.time()
    atomic_json(MANIFEST, manifest)
    return manifest


def set_drain() -> dict[str, Any]:
    control = {
        "mode": "DRAIN",
        "reason": "minimal_acquisition_gap_scope_reduction",
        "active_job_allowed_to_finish": "40319425",
        "updated_at": time.time(),
    }
    atomic_json(CONTROL, control)
    manifest = initialize()
    manifest["scheduler_mode"] = "DRAIN"
    manifest["drain_reason"] = control["reason"]
    atomic_json(MANIFEST, manifest)
    return control


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("init", "refresh", "dispatch", "watch", "drain"))
    parser.add_argument("--interval-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.action == "drain":
        value = set_drain()
    elif args.action == "init":
        value = initialize()
    elif args.action == "refresh":
        value = refresh(initialize())
    elif args.action == "dispatch":
        value = dispatch()
    else:
        while True:
            value = dispatch()
            active = {
                job["state"]
                for job in value["jobs"].values()
                if job["state"] in {"PENDING", "RUNNING", "RETRYING"}
            }
            print(json.dumps(value, indent=2), flush=True)
            if not active:
                return
            time.sleep(max(30, args.interval_seconds))
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
