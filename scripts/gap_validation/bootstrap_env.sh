#!/usr/bin/env bash
set -euo pipefail

export CAFD_ROOT="/blue/du.j/jinjiaguo/CAFD"
export HF_HOME="${CAFD_ROOT}/.cache/huggingface"
export UV_CACHE_DIR="${CAFD_ROOT}/.cache/uv"
cd "${CAFD_ROOT}"
module purge
module load conda/25.7.0 cuda/12.8.1

mkdir -p .cache/huggingface .cache/uv .cache/torch_extensions logs/gap_validation
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.11 .venv
fi
uv pip install --python .venv/bin/python torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
DS_BUILD_OPS=0 uv pip install --python .venv/bin/python -e '.[test,train]'
uv pip install --python .venv/bin/python -e vendor/trl --no-deps
.venv/bin/python -m gap_validation.artifact_init
uv pip freeze --python .venv/bin/python > artifacts/gap_validation/environment_freeze.txt
