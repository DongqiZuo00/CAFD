#!/usr/bin/env bash
set -euo pipefail
source /blue/du.j/jinjiaguo/CAFD/scripts/gap_validation/common.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc_per_node=4 -m gap_validation.teacher_responses \
  --model runs/qwen_olymmath_easy/teacher/selected_full_route/final_model \
  --output-dir runs/qwen_olymmath_easy/teacher_responses \
  --responses-per-prompt 8 \
  --required-successes 0 \
  --num-shards 4
python -m gap_validation.merge_teacher_responses \
  --input-dir runs/qwen_olymmath_easy/teacher_responses \
  --required-successes 6400
python -m gap_validation.prepare_sft_data \
  --teacher-successes runs/qwen_olymmath_easy/teacher_responses/teacher_successes.jsonl \
  --output-dir data/frozen/math_training/sft_streams
