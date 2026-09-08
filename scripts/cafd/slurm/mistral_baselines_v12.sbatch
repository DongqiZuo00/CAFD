#!/usr/bin/env bash
#SBATCH --job-name=cafd-baselines-v12
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/cafd/baselines-v12-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/cafd/baselines-v12-%j.err

set -euo pipefail
ROOT=/blue/du.j/jinjiaguo/CAFD
SOURCE_RUN="$ROOT/runs/cafd/experiments/mistral_cafd_disjoint_v7"
STATE_ROOT="$ROOT/state/cafd/acquisition_baselines_v12"
cd "$ROOT"
mkdir -p logs/cafd "$STATE_ROOT"
exec 9>"$STATE_ROOT/execution.lock"
if ! flock -n 9; then
  echo "another v12 baseline job owns the lock" >&2
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
export CAFD_CONFIG=configs/cafd/manufactoria_mistral_progressive_observation_v11.yaml

write_stage() {
  export BASELINE_STAGE=$1
  .venv/bin/python -c 'import json,os,pathlib,time;p=pathlib.Path(os.environ["STATE_ROOT"])/"job.json";t=p.with_suffix(".json.tmp");t.write_text(json.dumps({"stage":os.environ["BASELINE_STAGE"],"job_id":os.environ["SLURM_JOB_ID"],"time":time.time(),"frozen_test_accessed":False},indent=2,sort_keys=True)+"\n");os.replace(t,p)'
}
failure() {
  final_code=${1:-1}
  trap - ERR TERM INT
  export BASELINE_EXIT_CODE=$final_code
  .venv/bin/python -c 'import json,os,pathlib,time;p=pathlib.Path(os.environ["STATE_ROOT"])/"job.json";t=p.with_suffix(".json.tmp");t.write_text(json.dumps({"stage":"failed","exit_code":int(os.environ["BASELINE_EXIT_CODE"]),"job_id":os.environ["SLURM_JOB_ID"],"time":time.time(),"frozen_test_accessed":False},indent=2,sort_keys=True)+"\n");os.replace(t,p)' || true
  exit "$final_code"
}
trap 'failure $?' ERR
trap 'failure 143' TERM
trap 'failure 130' INT
: "${CAFD_EXPECTED_COMMIT:?missing CAFD_EXPECTED_COMMIT}"
test "$(git rev-parse HEAD)" = "$CAFD_EXPECTED_COMMIT"
git diff --quiet
git diff --cached --quiet

setup_run() {
  run_root="$ROOT/runs/cafd/experiments/$1"
  mkdir -p "$run_root/student_base"
  [ -e "$run_root/student_base/S0" ] || ln -s "$SOURCE_RUN/student_base/S0" "$run_root/student_base/S0"
  [ -e "$run_root/teacher" ] || ln -s "$SOURCE_RUN/teacher" "$run_root/teacher"
  test "$(readlink -f "$run_root/student_base/S0")" = "$(readlink -f "$SOURCE_RUN/student_base/S0")"
  test "$(readlink -f "$run_root/teacher")" = "$(readlink -f "$SOURCE_RUN/teacher")"
}

write_stage tests
.venv/bin/python -m pytest tests/cafd/test_mistral_acquisition_observation.py tests/cafd/test_mistral_generalization_v10.py tests/cafd/test_resume.py -q
setup_run mistral_direct_observation_v12
setup_run mistral_endpoint_observation_v12
write_stage direct_and_endpoint_training
(
  export CUDA_VISIBLE_DEVICES=0
  export CAFD_BASELINE_METHOD=direct
  .venv/bin/python -m cafd.mistral_baselines_v12 --root "$ROOT" --method direct
) >> "logs/cafd/direct-observation-v12-${SLURM_JOB_ID}.log" 2>&1 &
direct_pid=$!
(
  export CUDA_VISIBLE_DEVICES=1
  export CAFD_BASELINE_METHOD=endpoint
  .venv/bin/python -m cafd.mistral_baselines_v12 --root "$ROOT" --method endpoint
) >> "logs/cafd/endpoint-observation-v12-${SLURM_JOB_ID}.log" 2>&1 &
endpoint_pid=$!
status=0
wait "$direct_pid" || status=$?
wait "$endpoint_pid" || status=$?
test "$status" -eq 0
write_stage complete
