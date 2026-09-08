#!/usr/bin/env bash
set -euo pipefail
source /blue/du.j/jinjiaguo/CAFD/scripts/gap_validation/common.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m gap_validation.student_route \
  --method progressive_gkd \
  --teacher-route-root runs/qwen_olymmath_easy/teacher/selected_full_route \
  --root runs/qwen_olymmath_easy/progressive_gkd
