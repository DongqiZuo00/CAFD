#!/usr/bin/env bash
#SBATCH --job-name=cafd-gen-v10
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/cafd/mistral-gen-v10-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/cafd/mistral-gen-v10-%j.err

set -euo pipefail
ROOT=/blue/du.j/jinjiaguo/CAFD
SOURCE_RUN="$ROOT/runs/cafd/experiments/mistral_cafd_disjoint_v7"
STATE_ROOT="$ROOT/state/cafd/generalization_v10"
TEACHER_RUN=mistral_cafd_gen_v10_teacher
MIXED_RUN=mistral_cafd_gen_v10_mixed
cd "$ROOT"
mkdir -p logs/cafd "$STATE_ROOT"
exec 9>"$STATE_ROOT/execution.lock"
if ! flock -n 9; then
  echo "another CAFD v10 generalization audit owns the lock" >&2
  exit 75
fi

export PYTHONPATH="$ROOT:$ROOT/vendor/trl"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$ROOT/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/hub"
export TORCH_HOME="$ROOT/.cache/torch"
export XDG_CACHE_HOME="$ROOT/.cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export STATE_ROOT

write_stage() {
  export CAFD_STAGE=$1
  .venv/bin/python -c 'import json,os,pathlib,time; p=pathlib.Path(os.environ["STATE_ROOT"])/"job.json"; t=p.with_suffix(".json.tmp"); t.write_text(json.dumps({"stage":os.environ["CAFD_STAGE"],"job_id":os.environ["SLURM_JOB_ID"],"time":time.time(),"frozen_test_accessed":False},indent=2,sort_keys=True)+"\n",encoding="utf-8"); os.replace(t,p)'
}
failure() {
  final_code=${1:-1}
  trap - ERR TERM INT
  export CAFD_EXIT_CODE=$final_code
  .venv/bin/python -c 'import json,os,pathlib,time; p=pathlib.Path(os.environ["STATE_ROOT"])/"job.json"; t=p.with_suffix(".json.tmp"); t.write_text(json.dumps({"stage":"failed","exit_code":int(os.environ["CAFD_EXIT_CODE"]),"job_id":os.environ["SLURM_JOB_ID"],"time":time.time(),"frozen_test_accessed":False},indent=2,sort_keys=True)+"\n",encoding="utf-8"); os.replace(t,p)' || true
  exit "$final_code"
}
trap 'failure $?' ERR
trap 'failure 143' TERM
trap 'failure 130' INT

: "${CAFD_EXPECTED_COMMIT:?missing CAFD_EXPECTED_COMMIT}"
test "$(git rev-parse HEAD)" = "$CAFD_EXPECTED_COMMIT"
git diff --quiet
git diff --cached --quiet
write_stage tests
.venv/bin/python -m pytest   tests/cafd/test_mistral_generalization_v10.py   tests/cafd/test_mistral_support_full.py -q

write_stage freeze_nested_split
.venv/bin/python -c 'from pathlib import Path; from cafd.generalization_split import prepare; prepare(Path.cwd())'

prepare_run() {
  run_id=$1
  run_root="$ROOT/runs/cafd/experiments/$run_id"
  mkdir -p "$run_root/student_base"
  if [ ! -e "$run_root/student_base/S0" ]; then
    ln -s "$SOURCE_RUN/student_base/S0" "$run_root/student_base/S0"
  fi
  if [ ! -e "$run_root/teacher" ]; then
    ln -s "$SOURCE_RUN/teacher" "$run_root/teacher"
  fi
  test "$(readlink -f "$run_root/student_base/S0")" = "$(readlink -f "$SOURCE_RUN/student_base/S0")"
  test "$(readlink -f "$run_root/teacher")" = "$(readlink -f "$SOURCE_RUN/teacher")"
}
prepare_run "$TEACHER_RUN"
prepare_run "$MIXED_RUN"

write_stage parallel_cafd_training
(
  export CUDA_VISIBLE_DEVICES=0
  export CAFD_CONFIG=configs/cafd/manufactoria_mistral_cafd_gen_v10_teacher.yaml
  .venv/bin/python -m cafd.mistral_cafd_generalization_v10     --root "$ROOT" --method cafd
) >> "logs/cafd/mistral-gen-v10-teacher-${SLURM_JOB_ID}.log" 2>&1 &
teacher_pid=$!
(
  export CUDA_VISIBLE_DEVICES=1
  export CAFD_CONFIG=configs/cafd/manufactoria_mistral_cafd_gen_v10_mixed.yaml
  .venv/bin/python -m cafd.mistral_cafd_generalization_v10     --root "$ROOT" --method cafd
) >> "logs/cafd/mistral-gen-v10-mixed-${SLURM_JOB_ID}.log" 2>&1 &
mixed_pid=$!
status=0
wait "$teacher_pid" || status=$?
wait "$mixed_pid" || status=$?
test "$status" -eq 0

write_stage parallel_development_confirmation
(
  export CUDA_VISIBLE_DEVICES=0
  export CAFD_CONFIG=configs/cafd/manufactoria_mistral_cafd_gen_v10_teacher.yaml
  .venv/bin/python -m cafd.mistral_generalization_confirm --root "$ROOT"
) >> "logs/cafd/mistral-gen-v10-teacher-confirm-${SLURM_JOB_ID}.log" 2>&1 &
teacher_confirm_pid=$!
(
  export CUDA_VISIBLE_DEVICES=1
  export CAFD_CONFIG=configs/cafd/manufactoria_mistral_cafd_gen_v10_mixed.yaml
  .venv/bin/python -m cafd.mistral_generalization_confirm --root "$ROOT"
) >> "logs/cafd/mistral-gen-v10-mixed-confirm-${SLURM_JOB_ID}.log" 2>&1 &
mixed_confirm_pid=$!
status=0
wait "$teacher_confirm_pid" || status=$?
wait "$mixed_confirm_pid" || status=$?
test "$status" -eq 0

write_stage freeze_development_only_decision
.venv/bin/python -m cafd.mistral_generalization_decision --root "$ROOT"
write_stage complete
