#!/usr/bin/env bash
set -euo pipefail

export CAFD_ROOT="/blue/du.j/jinjiaguo/CAFD"
export HF_HOME="${CAFD_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_EXTENSIONS_DIR="${CAFD_ROOT}/.cache/torch_extensions"
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false

cd "${CAFD_ROOT}"
export TRITON_CACHE_DIR="${CAFD_ROOT}/.cache/triton/${SLURM_JOB_ID:-interactive}"
mkdir -p "${TRITON_CACHE_DIR}"
module purge
module load conda/25.7.0 cuda/12.8.1
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv; run scripts/gap_validation/bootstrap_env.sh first" >&2
  exit 2
fi
source .venv/bin/activate
printf 'SLURM_JOB_ID=%s CUDA_VISIBLE_DEVICES=%s SLURM_JOB_GPUS=%s\n' \
  "${SLURM_JOB_ID:-none}" "${CUDA_VISIBLE_DEVICES:-none}" "${SLURM_JOB_GPUS:-none}"
