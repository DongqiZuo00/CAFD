from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from .io import ROOT, atomic_json
from .minimal_gates import development_gates
from .minimal_runtime import ARTIFACT_ROOT, RUN_ID, RUN_ROOT, TEACHER_ROUTE, load_summary, write_run_status


STATE = ARTIFACT_ROOT / "scheduler_state.json"
LOG_ROOT = ROOT / "logs" / "gap_validation" / "minimal" / RUN_ID
SCRIPTS = {
    "teacher_recovery": ("minimal_teacher_recovery.sbatch", 192, "3-00:00:00"),
    "teacher_development": ("minimal_teacher_development.sbatch", 192, "2-00:00:00"),
    "student_direct_rlvr": ("minimal_direct.sbatch", 192, "3-00:00:00"),
    "student_oracle": ("minimal_oracle.sbatch", 192, "3-00:00:00"),
    "student_final_opd": ("minimal_opd.sbatch", 192, "3-00:00:00"),
    "frozen_final": ("minimal_final.sbatch", 192, "2-00:00:00"),
}


def command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def load_state() -> dict[str, Any]:
    if STATE.exists():
        return load_summary(STATE)
    return {"run_id": RUN_ID, "state": "RUNNING", "stages": {}, "created_at": time.time()}


def save_state(value: dict[str, Any]) -> None:
    value["updated_at"] = time.time()
    atomic_json(STATE, value)


def requested_b200_gpus() -> int:
    user = os.environ.get("USER", "jinjiaguo")
    result = command(["squeue", "-h", "-u", user, "-p", "hpg-b200", "-o", "%b"])
    total = 0
    for line in result.stdout.splitlines():
        match = re.search(r"(?:gpu|gres/gpu)(?::b200)?:(\d+)", line)
        if match:
            total += int(match.group(1))
    return total


def wait_for_gpu_budget(required: int = 4) -> None:
    while requested_b200_gpus() + required > 4:
        time.sleep(60)


def slurm_state(job_id: str) -> tuple[str, str]:
    result = command(["sacct", "-n", "-X", "-j", job_id, "-o", "State,ExitCode", "-P"])
    line = next((line for line in result.stdout.splitlines() if line.strip()), "")
    if not line:
        return "UNKNOWN", ""
    parts = line.split("|")
    return parts[0].split()[0], parts[1] if len(parts) > 1 else ""


def submit_stage(stage: str, max_retries: int = 3) -> str:
    state = load_state()
    stage_state = state["stages"].setdefault(stage, {"attempts": []})
    if stage_state.get("state") == "COMPLETED":
        return str(stage_state["job_id"])
    script_name, memory_gb, time_limit = SCRIPTS[stage]
    for _ in range(max_retries - len(stage_state["attempts"])):
        wait_for_gpu_budget(4)
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        result = command(
            [
                "sbatch", "--parsable", "--partition=hpg-b200", "--account=du.j", "--qos=du.j",
                f"--job-name=minimal-{stage}", "--gpus=4", "--cpus-per-task=56",
                f"--mem={memory_gb}gb", f"--time={time_limit}",
                f"--output={LOG_ROOT}/{stage}-%j.out", f"--error={LOG_ROOT}/{stage}-%j.err",
                str(ROOT / "scripts" / "gap_validation" / "slurm" / script_name),
            ]
        )
        if result.returncode:
            stage_state["attempts"].append(
                {"submit_error": result.stderr.strip(), "time": time.time()}
            )
            save_state(state)
            time.sleep(60)
            continue
        job_id = result.stdout.strip().split(";", 1)[0]
        stage_state["job_id"] = job_id
        stage_state["state"] = "RUNNING"
        stage_state["attempts"].append({"job_id": job_id, "submitted_at": time.time()})
        save_state(state)
        while True:
            current, exit_code = slurm_state(job_id)
            if current == "COMPLETED":
                stage_state["state"] = "COMPLETED"
                stage_state["completed_at"] = time.time()
                save_state(state)
                return job_id
            if current in {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED", "NODE_FAIL"}:
                stage_state["state"] = "RETRYING"
                stage_state["attempts"][-1].update(
                    {"terminal_state": current, "exit_code": exit_code, "ended_at": time.time()}
                )
                save_state(state)
                break
            time.sleep(60)
    stage_state["state"] = "FAILED"
    save_state(state)
    raise RuntimeError(f"stage failed after {max_retries} attempts: {stage}")


def terminal(reason: str, statement: str, gates: dict[str, Any] | None = None) -> None:
    state = load_state()
    state["state"] = reason
    state["statement"] = statement
    if gates is not None:
        state["development_gates"] = gates
    save_state(state)
    write_run_status("terminal", reason, statement=statement, development_gates=gates)


def main() -> None:
    write_run_status("legacy_teacher", "VALIDATING")
    if not (TEACHER_ROUTE / "run_summary.json").exists():
        submit_stage("teacher_recovery")
    if not (TEACHER_ROUTE / "run_summary.json").exists():
        raise RuntimeError("T400 route remains incomplete after automatic recovery")
    submit_stage("teacher_development")
    teacher_gate = load_summary(ARTIFACT_ROOT / "development" / "teacher_gate.json")
    t0 = int(teacher_gate["teacher_t0_correct"])
    t400 = int(teacher_gate["teacher_t400_correct"])
    s0 = int(teacher_gate["student_s0"]["correct"])
    gates = development_gates(t0=t0, t400=t400, s0=s0)
    if not gates["teacher_acquisition"]:
        terminal(
            "NO_TEACHER_ACQUISITION",
            "Teacher acquisition was not established; therefore this setting cannot test the intended acquisition gap.",
            gates,
        )
        return

    submit_stage("student_direct_rlvr")
    direct = load_summary(RUN_ROOT / "student_direct_rlvr" / "checkpoint_selection.json")
    gates = development_gates(t0=t0, t400=t400, s0=s0, direct=int(direct["development_correct"]))
    if not gates["direct_failure"]:
        terminal(
            "NO_STUDENT_REDISCOVERY_GAP",
            "Direct Student RLVR reached or exceeded T400 on development; the required Student rediscovery gap is absent.",
            gates,
        )
        return

    submit_stage("student_oracle")
    oracle = load_summary(RUN_ROOT / "student_oracle" / "checkpoint_selection.json")
    gates = development_gates(
        t0=t0, t400=t400, s0=s0, direct=int(direct["development_correct"]),
        oracle=int(oracle["development_correct"]),
    )
    if not gates["oracle_capacity"]:
        terminal(
            "STUDENT_CAPACITY_NOT_ESTABLISHED",
            "Student capacity was not established; the observed difference cannot be identified as an acquisition gap.",
            gates,
        )
        return

    submit_stage("student_final_opd")
    opd = load_summary(RUN_ROOT / "student_final_opd" / "checkpoint_selection.json")
    gates = development_gates(
        t0=t0, t400=t400, s0=s0, direct=int(direct["development_correct"]),
        oracle=int(oracle["development_correct"]), opd=int(opd["development_correct"]),
    )
    if not gates["final_open"]:
        terminal(
            "DEVELOPMENT_PATTERN_FALSIFIED",
            "An acquisition gap was not established in this setting; the preregistered development gate did not open the frozen final split.",
            gates,
        )
        return


    submit_stage("frozen_final")
    result = load_summary(ARTIFACT_ROOT / "final_result.json")
    state = load_state()
    state["state"] = "COMPLETED"
    state["final_result"] = str(ARTIFACT_ROOT / "final_result.json")
    state["decision"] = result["decision"]
    save_state(state)
    write_run_status("frozen_final", "COMPLETED", decision=result["decision"])


if __name__ == "__main__":
    main()
