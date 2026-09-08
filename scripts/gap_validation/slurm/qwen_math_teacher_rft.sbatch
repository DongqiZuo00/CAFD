#!/usr/bin/env bash
set -euo pipefail
source /blue/du.j/jinjiaguo/CAFD/scripts/gap_validation/common.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m gap_validation.sft_route \
  --method teacher_rft \
  --dataset data/frozen/math_training/sft_streams/rft_stream.jsonl \
  --root runs/qwen_olymmath_easy/teacher_rft
