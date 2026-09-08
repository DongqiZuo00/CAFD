from __future__ import annotations

import json

from .scheduler import initialize, refresh


def main() -> None:
    manifest = refresh(initialize())
    print(f"{'JOB':38} {'STATE':24} {'GPUS':>4} {'SLURM':>10}")
    for name, job in manifest["jobs"].items():
        print(f"{name:38} {job['state']:24} {job.get('gpus', 0):4d} {str(job.get('slurm_job_id', '-')):>10}")
        if job.get("blocker"):
            print(f"  blocker: {job['blocker']}")


if __name__ == "__main__":
    main()

