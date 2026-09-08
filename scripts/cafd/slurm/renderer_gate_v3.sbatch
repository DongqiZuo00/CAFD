#!/usr/bin/env bash
#SBATCH --job-name=cafd-renderer-gate-v3
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/cafd/renderer-gate-v3-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/cafd/renderer-gate-v3-%j.err

set -euo pipefail

ROOT=/blue/du.j/jinjiaguo/CAFD
cd "$ROOT"
mkdir -p logs/cafd state/cafd artifacts/cafd .cache/huggingface
export PYTHONPATH="$ROOT:$ROOT/vendor/trl"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$ROOT/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/hub"
export TORCH_HOME="$ROOT/.cache/torch"
export XDG_CACHE_HOME="$ROOT/.cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

.venv/bin/python -c "import pathlib,yaml; c=yaml.safe_load(pathlib.Path('configs/cafd/manufactoria_has.yaml').read_text()); assert c['protocol']=='compute_bounded_v3_chat_hierarchical'; assert c['generation']['prompt_renderer']=='qwen3_instruct_no_thinking'; assert c['reward']['mode']=='hierarchical'"
.venv/bin/python -m cafd.renderer_gate --root "$ROOT"
