#!/usr/bin/env bash
set -euo pipefail

ROOT=/blue/du.j/jinjiaguo/CAFD
cd "$ROOT"
.venv/bin/python -c "import json,pathlib,yaml; p=pathlib.Path('archive/cafd_invalid_48x16x8192/BUDGET_INVALID.json'); d=json.loads(p.read_text()); assert d['resume_forbidden'] is True; c=yaml.safe_load(pathlib.Path('configs/cafd/manufactoria_has.yaml').read_text()); assert c['protocol']=='compute_bounded_v2'"
for gate in runs/cafd/capacity/gate.json runs/cafd/teacher/gate.json; do
  if [[ -f "$gate" ]] && grep -Eq 'CAPACITY_BLOCKED|TEACHER_ROUTE_FAILED' "$gate"; then
    cat "$gate"
    exit 1
  fi
done
active=$(squeue -h -u jinjiaguo -n cafd-cb -o '%A' | head -n 1 || true)
if [[ -n "$active" ]]; then
  printf '%s\n' "$active"
  exit 0
fi
job_id=$(sbatch --parsable scripts/cafd/slurm/cafd_v1.sbatch)
mkdir -p state/cafd
printf '%s\n' "$job_id" > state/cafd/job_id
printf '%s\n' "$job_id"
