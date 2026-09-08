#!/usr/bin/env bash
#SBATCH --job-name=cafd-direction-v15
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/cafd/direction-v15-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/cafd/direction-v15-%j.err
set -euo pipefail
cd /blue/du.j/jinjiaguo/CAFD
export PYTHONPATH="$PWD:$PWD/vendor/trl"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p state/cafd/experiments/mistral_cafd_direction_v15
exec 9>state/cafd/experiments/mistral_cafd_direction_v15/execution.lock
flock -n 9 || exit 75
.venv/bin/python -m pytest tests/cafd/test_direction_v15.py -q
.venv/bin/python -m cafd.mistral_direction_v15
