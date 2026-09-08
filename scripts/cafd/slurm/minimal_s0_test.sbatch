#!/usr/bin/env bash
#SBATCH --job-name=cafd-minimal-s0-test
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/cafd/minimal-s0-test-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/cafd/minimal-s0-test-%j.err
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


.venv/bin/python -u -m cafd.minimal_baseline_finalize --run-id mistral_cafd_minimal_s0_s2027
.venv/bin/python -u -m cafd.minimal_baseline_finalize --run-id mistral_cafd_minimal_s0_s2027
