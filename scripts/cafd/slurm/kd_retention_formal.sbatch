#!/usr/bin/env bash
#SBATCH --job-name=cafd-kd-retention-formal
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/cafd/kd-retention-formal-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/cafd/kd-retention-formal-%j.err
set -euo pipefail
cd /blue/du.j/jinjiaguo/CAFD
export PYTHONPATH="$PWD:$PWD/vendor/trl"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TMPDIR="$PWD/tmp"
mkdir -p state/cafd/experiments/mistral_cafd_kd_retention_replay_s2027
exec 9>state/cafd/experiments/mistral_cafd_kd_retention_replay_s2027/execution.lock
flock -n 9 || exit 75
.venv/bin/python - <<'PY'
import json
from pathlib import Path
root=Path.cwd()
run=root/'runs/cafd/experiments/mistral_cafd_kd_retention_replay_s2027_smoke'
complete=json.loads((run/'complete.json').read_text())
resume=json.loads((run/'resume_validation.json').read_text())
assert complete['status']=='SMOKE_PASSED' and complete['rounds']==complete['actual_updates']==2
assert resume['loaded_round']==resume['actual_updates']==1
assert resume['optimizer_fp32_validated'] and resume['distinct_roles']
print('SMOKE_AND_RESUME_GATE_PASSED',flush=True)
PY
RESUME=()
if [[ -f runs/cafd/experiments/mistral_cafd_kd_retention_replay_s2027/resume.pt ]]; then RESUME=(--resume); fi
.venv/bin/python -u -m cafd.kd_retention_train --profile formal --run-id mistral_cafd_kd_retention_replay_s2027 "${RESUME[@]}"
.venv/bin/python -u -m cafd.kd_retention_finalize --run-id mistral_cafd_kd_retention_replay_s2027
.venv/bin/python -u -m cafd.kd_retention_finalize --run-id mistral_cafd_kd_retention_replay_s2027

.venv/bin/python -u -m cafd.kd_retention_report
