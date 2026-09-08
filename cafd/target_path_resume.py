"""Recover stale diagnostic claims only under the inherited run-wide flock.

Finished prior-job claim directories are archived, never deleted. Rollouts,
costs, checkpoints, and result files are untouched. Unknown/active Slurm state,
missing ownership, same-job claims, and unsafe paths fail closed before moving
anything. Per-policy identity validation remains the generation runner's job.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path("/blue/du.j/jinjiaguo/CAFD")
RUN = ROOT/"runs/cafd/experiments/mistral_target_path_t1t2_v1"
TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def numeric_job(value):
    text = str(value)
    if re.fullmatch(r"[1-9][0-9]*", text) is None:
        raise RuntimeError(f"ambiguous Slurm job identity: {value!r}")
    return text


def root_job_state(job_id, sacct_output):
    """Ignore step records: only the exact allocation's final state qualifies."""
    job_id = numeric_job(job_id)
    matches = []
    for line in sacct_output.splitlines():
        fields = [x.strip() for x in line.split("|")]
        if len(fields) >= 2 and fields[0] == job_id:
            state = fields[1].split()[0].rstrip("+") if fields[1] else ""
            matches.append(state)
    if len(matches) != 1:
        raise RuntimeError(f"sacct must return exactly one root record for {job_id}: {matches}")
    return matches[0]


def query_slurm_state(job_id):
    result = subprocess.run(["sacct", "-j", numeric_job(job_id), "-n", "-P",
                             "--format=JobIDRaw,State"], cwd=ROOT,
                            text=True, capture_output=True, timeout=30, check=True)
    return root_job_state(job_id, result.stdout)


def archive_stale_claims(run_dir, current_job, query_state):
    """Validate all claims before archiving any; caller must hold exclusive lock."""
    run_dir, current_job = Path(run_dir).resolve(), numeric_job(current_job)
    claims = run_dir/"claims"
    if not claims.exists():
        return dict(status="no_claims", archived=[], current_job=current_job)
    if claims.is_symlink() or claims.resolve().parent != run_dir:
        raise RuntimeError("unsafe claims directory")
    candidates, states = [], {}
    for claim in sorted(claims.iterdir()):
        if not claim.is_dir() or claim.is_symlink() or claim.resolve().parent != claims:
            raise RuntimeError(f"unexpected or unsafe claim path: {claim}")
        owner_path = claim/"owner.json"
        if owner_path.is_symlink() or not owner_path.is_file():
            raise RuntimeError(f"missing/unsafe claim owner: {claim.name}")
        owner = json.loads(owner_path.read_text())
        previous_job = numeric_job(owner.get("job"))
        if previous_job == current_job:
            raise RuntimeError(f"same-job claim may still be active: {claim.name}; refusing takeover")
        if previous_job not in states:
            states[previous_job] = str(query_state(previous_job))
        if states[previous_job] not in TERMINAL:
            raise RuntimeError(f"previous job {previous_job} is not proven terminal: {states[previous_job]!r}")
        candidates.append((claim, owner, previous_job))
    if not candidates:
        return dict(status="no_claims", archived=[], current_job=current_job)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive_root = run_dir/"claim_archive"
    if archive_root.is_symlink():
        raise RuntimeError("unsafe claim archive root")
    archive = archive_root/f"before_job_{current_job}_{timestamp}"
    if not archive.resolve().is_relative_to(run_dir):
        raise RuntimeError("archive target escapes diagnostic run")
    archive.mkdir(parents=True, exist_ok=False)
    worker_provenance=archive/"worker_provenance"
    worker_provenance.mkdir()
    copied=[]
    for source in sorted(run_dir.glob("worker*_state.json"))+[run_dir/"worker_exit.json"]:
        if not source.exists():
            continue
        if source.is_symlink() or not source.is_file() or source.resolve().parent != run_dir:
            raise RuntimeError(f"unsafe worker provenance path: {source}")
        destination=worker_provenance/source.name
        shutil.copy2(source,destination)
        copied.append(dict(source=str(source),preserved_copy=str(destination)))
    archived = []
    for claim, owner, previous_job in candidates:
        destination = archive/claim.name
        if claim.resolve().parent != claims or destination.resolve().parent != archive:
            raise RuntimeError("claim/archive identity changed during recovery")
        claim.rename(destination)
        archived.append(dict(policy=claim.name, previous_job=previous_job,
                             previous_state=states[previous_job], owner=owner,
                             preserved_claim_directory=str(destination)))
    report = dict(status="archived_terminal_job_claims", current_job=current_job,
                  archived=archived, prior_jobs=states, worker_provenance_copies=copied,
                  rollouts_modified=False, costs_modified=False, checkpoints_modified=False)
    (archive/"recovery.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def verify_inherited_lock(run_dir, fd):
    import fcntl
    expected = (Path(run_dir)/"execution.lock").stat()
    inherited = os.fstat(fd)
    if (expected.st_dev, expected.st_ino) != (inherited.st_dev, inherited.st_ino):
        raise RuntimeError("inherited descriptor is not this run's execution.lock")
    # Same inherited open-file description succeeds; another owner's lock fails.
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID"))
    parser.add_argument("--lock-fd", type=int, default=9)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT or args.run_dir.resolve() != RUN:
        raise RuntimeError("claim recovery is restricted to the exact CAFD diagnostic run")
    job = numeric_job(args.job_id)
    if job != numeric_job(os.environ.get("SLURM_JOB_ID")):
        raise RuntimeError("recovery job differs from actual Slurm allocation")
    verify_inherited_lock(args.run_dir, args.lock_fd)
    print(json.dumps(archive_stale_claims(args.run_dir, job, query_slurm_state)), flush=True)


if __name__ == "__main__":
    main()
