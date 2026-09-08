#!/usr/bin/env bash
set -euo pipefail
source /blue/du.j/jinjiaguo/CAFD/scripts/gap_validation/common.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m gap_validation.sft_route \
  --method capacity_control \
  --dataset data/frozen/math_training/sft_streams/capacity_stream.jsonl \
  --root runs/qwen_olymmath_easy/capacity_control
