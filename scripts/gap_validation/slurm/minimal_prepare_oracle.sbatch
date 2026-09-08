#!/usr/bin/env bash
#SBATCH --job-name=minimal-oracle-data
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=02:00:00
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/gap_validation/minimal/oracle-data-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/gap_validation/minimal/oracle-data-%j.err
set -euo pipefail
source /blue/du.j/jinjiaguo/CAFD/scripts/gap_validation/common.sh
python -m gap_validation.minimal_prepare_oracle \
  --input data/frozen/math_training/student_prompt_stream.jsonl \
  --output runs/qwen_olymmath_easy_minimal_gap_seed42_20260826/student_oracle/data/oracle_stream.jsonl \
  --status runs/qwen_olymmath_easy_minimal_gap_seed42_20260826/student_oracle/data/status.json \
  --failure-output runs/qwen_olymmath_easy_minimal_gap_seed42_20260826/student_oracle/data/canonical_failures.jsonl \
  --max-updates 800 \
  --sequences-per-update 32
