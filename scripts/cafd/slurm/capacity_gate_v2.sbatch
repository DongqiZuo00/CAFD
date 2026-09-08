#!/usr/bin/env bash
#SBATCH --job-name=cafd-capgate
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/cafd/capacity-gate-v2-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/cafd/capacity-gate-v2-%j.err

set -euo pipefail

ROOT=/blue/du.j/jinjiaguo/CAFD
cd "$ROOT"
mkdir -p logs/cafd state/cafd runs/cafd artifacts/cafd .cache/huggingface
export PYTHONPATH="$ROOT:$ROOT/vendor/trl${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$ROOT/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/hub"
export TORCH_HOME="$ROOT/.cache/torch"
export XDG_CACHE_HOME="$ROOT/.cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

.venv/bin/torchrun --standalone --nproc_per_node=4 -m cafd.train_capacity --root "$ROOT"
