#!/usr/bin/env bash
#SBATCH --job-name=cafd-anchored-v14
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/cafd/anchored-v14-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/cafd/anchored-v14-%j.err
set -euo pipefail
cd /blue/du.j/jinjiaguo/CAFD
ROOT=/blue/du.j/jinjiaguo/CAFD
mkdir -p state/cafd/experiments/mistral_cafd_anchored_v14
exec 9>state/cafd/experiments/mistral_cafd_anchored_v14/execution.lock
flock -n 9 || exit 75
export PYTHONPATH="$ROOT:$ROOT/vendor/trl"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$ROOT/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/hub"
export TORCH_HOME="$ROOT/.cache/torch"
export XDG_CACHE_HOME="$ROOT/.cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export CAFD_CONFIG=configs/cafd/manufactoria_mistral_cafd_gen_v10_mixed.yaml
.venv/bin/python -m pytest -p no:cacheprovider tests/cafd/test_anchored_target.py -q
.venv/bin/python -m cafd.mistral_anchored_v14 --root "$ROOT"
