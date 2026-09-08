#!/usr/bin/env bash
#SBATCH --job-name=cafd-mpc-v1-formal
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/cafd/mpc-v1-formal-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/cafd/mpc-v1-formal-%j.err
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
mkdir -p state/cafd/experiments/mistral_cafd_mpc_v1_formal
exec 9>state/cafd/experiments/mistral_cafd_mpc_v1_formal/execution.lock
flock -n 9 || exit 75
.venv/bin/python -m pytest tests/cafd/test_mpc_*.py -q
.venv/bin/python -u -m cafd.mpc_train --profile formal --run-id mistral_cafd_mpc_v1_formal "$@"
.venv/bin/python -u -m cafd.mpc_finalize --run-id mistral_cafd_mpc_v1_formal
