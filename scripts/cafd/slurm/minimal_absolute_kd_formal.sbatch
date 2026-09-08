#!/usr/bin/env bash
#SBATCH --job-name=cafd-minimal-absolute_kd-formal
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/cafd/minimal-absolute_kd-formal-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/cafd/minimal-absolute_kd-formal-%j.err
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
export TRITON_CACHE_DIR="$PWD/.cache/triton"
export XDG_CACHE_HOME="$PWD/.cache"
exec 9>state/cafd/minimal_baselines_s2027/execution.lock
flock -n 9 || exit 75


.venv/bin/python reports/minimal_baselines_s2027/validate_readiness.py
RESUME=()
if [[ -f runs/cafd/experiments/mistral_cafd_minimal_absolute_kd_s2027/resume.pt ]]; then RESUME=(--resume); fi
.venv/bin/python -u -m cafd.minimal_baseline_train --condition absolute_kd --profile formal "${RESUME[@]}"
.venv/bin/python -u -m cafd.minimal_baseline_finalize --run-id mistral_cafd_minimal_absolute_kd_s2027
.venv/bin/python -u -m cafd.minimal_baseline_finalize --run-id mistral_cafd_minimal_absolute_kd_s2027
