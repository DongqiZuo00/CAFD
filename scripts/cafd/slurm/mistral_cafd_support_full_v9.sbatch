#!/usr/bin/env bash
#SBATCH --job-name=cafd-mistral-v9-full
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/cafd/mistral-cafd-v9-full-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/cafd/mistral-cafd-v9-full-%j.err

set -euo pipefail
ROOT=/blue/du.j/jinjiaguo/CAFD
RUN_ID=mistral_cafd_support_full_v9
SOURCE_RUN="$ROOT/runs/cafd/experiments/mistral_cafd_disjoint_v7"
RUN_ROOT="$ROOT/runs/cafd/experiments/$RUN_ID"
STATE_ROOT="$ROOT/state/cafd/experiments/$RUN_ID"
cd "$ROOT"
mkdir -p logs/cafd "$STATE_ROOT" "$RUN_ROOT/student_base"
exec 9>"$STATE_ROOT/execution.lock"
if ! flock -n 9; then
  echo "another Mistral v9 full CAFD execution owns the lock" >&2
  exit 75
fi
export CAFD_CONFIG=configs/cafd/manufactoria_mistral_cafd_support_full_v9.yaml
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
  export CAFD_MISTRAL_STAGE=$1
  .venv/bin/python -c 'import json,os,pathlib,time; p=pathlib.Path(os.environ["STATE_ROOT"])/"full_job.json"; t=p.with_suffix(".json.tmp"); t.write_text(json.dumps({"stage":os.environ["CAFD_MISTRAL_STAGE"],"job_id":os.environ["SLURM_JOB_ID"],"time":time.time()},indent=2,sort_keys=True)+"\n",encoding="utf-8"); os.replace(t,p)'
}
failure() {
  final_code=${1:-1}
  trap - ERR TERM INT
  export CAFD_MISTRAL_EXIT_CODE=$final_code
  .venv/bin/python -c 'import json,os,pathlib,time; p=pathlib.Path(os.environ["STATE_ROOT"])/"full_job.json"; t=p.with_suffix(".json.tmp"); t.write_text(json.dumps({"stage":"failed","exit_code":int(os.environ["CAFD_MISTRAL_EXIT_CODE"]),"job_id":os.environ["SLURM_JOB_ID"],"time":time.time()},indent=2,sort_keys=True)+"\n",encoding="utf-8"); os.replace(t,p)' || true
  exit "$final_code"
}
trap 'failure $?' ERR
trap 'failure 143' TERM
trap 'failure 130' INT

: "${CAFD_EXPECTED_COMMIT:?missing CAFD_EXPECTED_COMMIT}"
test "$(git rev-parse HEAD)" = "$CAFD_EXPECTED_COMMIT"
git diff --quiet
git diff --cached --quiet
if grep -qi qwen "$CAFD_CONFIG"; then
  echo "QWEN_BANNED: active config contains Qwen" >&2
  exit 64
fi
.venv/bin/python -c 'import json; r=json.load(open("runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher/route.json")); assert r["status"]=="frozen" and r["kind"]=="mistral_full_acquisition_raw_instruct_to_sft_to_rl" and len(r["checkpoints"])==6'

test -d "$SOURCE_RUN/student_base/S0"
test -f "$SOURCE_RUN/teacher/route.json"
if [ ! -e "$RUN_ROOT/student_base/S0" ]; then
  ln -s "$SOURCE_RUN/student_base/S0" "$RUN_ROOT/student_base/S0"
fi
if [ ! -e "$RUN_ROOT/teacher" ]; then
  ln -s "$SOURCE_RUN/teacher" "$RUN_ROOT/teacher"
fi
test "$(readlink -f "$RUN_ROOT/student_base/S0")" = "$(readlink -f "$SOURCE_RUN/student_base/S0")"
test "$(readlink -f "$RUN_ROOT/teacher")" = "$(readlink -f "$SOURCE_RUN/teacher")"

write_stage tests
.venv/bin/python -m pytest tests/cafd/test_mistral_support_full.py tests/cafd/test_resume.py -q

write_stage cafd_training
.venv/bin/python -m cafd.mistral_cafd_support_full --root "$ROOT" --method cafd >> "logs/cafd/mistral-cafd-support-full-v9-${SLURM_JOB_ID}.log" 2>&1

write_stage freeze_development_selection
.venv/bin/python -m cafd.mistral_cafd_only_artifacts --root "$ROOT" --mode freeze

write_stage frozen_test
.venv/bin/python -m cafd.mistral_evaluate_condition --root "$ROOT" --condition cafd >> "logs/cafd/mistral-cafd-final-v9-${SLURM_JOB_ID}.log" 2>&1

write_stage finalize
.venv/bin/python -m cafd.mistral_cafd_only_artifacts --root "$ROOT" --mode finalize
write_stage complete
