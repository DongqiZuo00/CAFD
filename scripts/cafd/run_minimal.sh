#!/usr/bin/env bash
set -euo pipefail

ROOT=/blue/du.j/jinjiaguo/CAFD
cd "$ROOT"
mkdir -p logs/cafd state/cafd runs/cafd artifacts/cafd vendor
if [[ ! -d vendor/rl-grok-recipe/.git ]]; then
  git clone https://github.com/sunblaze-ucb/rl-grok-recipe vendor/rl-grok-recipe
fi
git -C vendor/rl-grok-recipe fetch --quiet origin 8500bec984d4a84a4aa94ca3adc31c004aa6a388
git -C vendor/rl-grok-recipe checkout --quiet 8500bec984d4a84a4aa94ca3adc31c004aa6a388

export PYTHONPATH="$ROOT:$ROOT/vendor/trl${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -c "import json,pathlib,yaml; p=pathlib.Path('archive/cafd_invalid_48x16x8192/BUDGET_INVALID.json'); d=json.loads(p.read_text()); assert d['status']=='BUDGET_INVALID' and d['resume_forbidden'] is True; c=yaml.safe_load(pathlib.Path('configs/cafd/manufactoria_has.yaml').read_text()); assert c['protocol']=='compute_bounded_v2'; assert c['teacher_route']['prompts_per_update']==4 and c['teacher_route']['rollouts_per_prompt']==8"
.venv/bin/python -m pytest tests/cafd -q

active=$(squeue -h -u jinjiaguo -n cafd-cb -o '%A' | head -n 1 || true)
if [[ -n "$active" ]]; then
  printf '%s\n' "$active"
  exit 0
fi
job_id=$(sbatch --parsable scripts/cafd/slurm/cafd_v1.sbatch)
printf '%s\n' "$job_id" > state/cafd/job_id
printf '%s\n' "$job_id"
