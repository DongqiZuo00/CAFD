#!/usr/bin/env bash
set -euo pipefail

ROOT=/blue/du.j/jinjiaguo/CAFD
cd "$ROOT"
job_id=$(cat state/cafd/job_id 2>/dev/null || true)
if [[ -n "$job_id" ]]; then
  squeue -j "$job_id" -o '%.18i %.24j %.9T %.10M %.19S %.20R %b' || true
  sstat -j "${job_id}.batch" --format=JobID,AveCPU,AveRSS,MaxRSS -n 2>/dev/null || true
fi
for file in state/cafd/teacher.json state/cafd/capacity.json state/cafd/direct.json state/cafd/endpoint.json state/cafd/progressive.json state/cafd/cafd.json state/cafd/freeze.json state/cafd/job.json; do
  if [[ -f "$file" ]]; then
    printf '%s\n' "--- $file"
    cat "$file"
  fi
done
for file in logs/cafd/slurm-*.err logs/cafd/teacher.log logs/cafd/capacity.log logs/cafd/direct.log logs/cafd/endpoint.log logs/cafd/progressive.log logs/cafd/cafd.log; do
  if [[ -f "$file" ]]; then
    printf '%s\n' "--- $file"
    tail -n 12 "$file"
  fi
done
