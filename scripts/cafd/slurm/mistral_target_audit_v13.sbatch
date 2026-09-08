#!/usr/bin/env bash
#SBATCH --job-name=cafd-target-audit-v13
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/cafd/target-audit-v13-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/cafd/target-audit-v13-%j.err

set -euo pipefail
ROOT=/blue/du.j/jinjiaguo/CAFD
STATE_ROOT="$ROOT/state/cafd/experiments/mistral_cafd_target_audit_v13"
cd "$ROOT"
mkdir -p "$STATE_ROOT" logs/cafd
exec 9>"$STATE_ROOT/execution.lock"
if ! flock -n 9; then
  echo "another target audit owns the lock" >&2
  exit 75
fi

export PYTHONPATH="$ROOT:$ROOT/vendor/trl"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$ROOT/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/hub"
export TORCH_HOME="$ROOT/.cache/torch"
export XDG_CACHE_HOME="$ROOT/.cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

.venv/bin/python -m pytest tests/cafd/test_relative_target.py -q
.venv/bin/python -m cafd.mistral_target_audit_v13 \
  --root "$ROOT" \
  --sequences-per-phase 2 \
  --positions-per-sequence 64
