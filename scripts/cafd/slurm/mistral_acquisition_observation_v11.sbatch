#!/usr/bin/env bash
#SBATCH --job-name=cafd-observe-v11
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/cafd/acquisition-observe-v11-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/cafd/acquisition-observe-v11-%j.err

set -euo pipefail
ROOT=/blue/du.j/jinjiaguo/CAFD
SOURCE_RUN="$ROOT/runs/cafd/experiments/mistral_cafd_disjoint_v7"
CAFD_RUN=mistral_cafd_gen_v10_mixed
GKD_RUN=mistral_progressive_gkd_observation_v11
STATE_ROOT="$ROOT/state/cafd/acquisition_observation_v11"
cd "$ROOT"
mkdir -p logs/cafd "$STATE_ROOT"
exec 9>"$STATE_ROOT/execution.lock"
if ! flock -n 9; then
  echo "another acquisition observation owns the lock" >&2
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
  export OBSERVATION_STAGE=$1
  .venv/bin/python -c 'import json,os,pathlib,time; p=pathlib.Path(os.environ["STATE_ROOT"])/"job.json"; t=p.with_suffix(".json.tmp"); t.write_text(json.dumps({"stage":os.environ["OBSERVATION_STAGE"],"job_id":os.environ["SLURM_JOB_ID"],"time":time.time(),"frozen_test_accessed":False},indent=2,sort_keys=True)+"\n",encoding="utf-8"); os.replace(t,p)'
}
failure() {
  final_code=${1:-1}
  trap - ERR TERM INT
  export OBSERVATION_EXIT_CODE=$final_code
  .venv/bin/python -c 'import json,os,pathlib,time; p=pathlib.Path(os.environ["STATE_ROOT"])/"job.json"; t=p.with_suffix(".json.tmp"); t.write_text(json.dumps({"stage":"failed","exit_code":int(os.environ["OBSERVATION_EXIT_CODE"]),"job_id":os.environ["SLURM_JOB_ID"],"time":time.time(),"frozen_test_accessed":False},indent=2,sort_keys=True)+"\n",encoding="utf-8"); os.replace(t,p)' || true
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
.venv/bin/python -m pytest   tests/cafd/test_mistral_acquisition_observation.py   tests/cafd/test_mistral_generalization_v10.py   tests/cafd/test_mistral_support_full.py -q
.venv/bin/python -c 'from pathlib import Path; from cafd.generalization_split import prepare; prepare(Path.cwd())'

gkd_root="$ROOT/runs/cafd/experiments/$GKD_RUN"
mkdir -p "$gkd_root/student_base"
if [ ! -e "$gkd_root/student_base/S0" ]; then
  ln -s "$SOURCE_RUN/student_base/S0" "$gkd_root/student_base/S0"
fi
if [ ! -e "$gkd_root/teacher" ]; then
  ln -s "$SOURCE_RUN/teacher" "$gkd_root/teacher"
fi
test "$(readlink -f "$gkd_root/student_base/S0")" = "$(readlink -f "$SOURCE_RUN/student_base/S0")"
test "$(readlink -f "$gkd_root/teacher")" = "$(readlink -f "$SOURCE_RUN/teacher")"
test -f "$ROOT/runs/cafd/experiments/$CAFD_RUN/cafd/selected.json"

write_stage progressive_training_and_cafd_milestone_observation
(
  export CUDA_VISIBLE_DEVICES=0
  export CAFD_CONFIG=configs/cafd/manufactoria_mistral_progressive_observation_v11.yaml
  .venv/bin/python -m cafd.mistral_progressive_observation_v11     --root "$ROOT" --method progressive
) >> "logs/cafd/progressive-observation-v11-${SLURM_JOB_ID}.log" 2>&1 &
gkd_pid=$!
(
  export CUDA_VISIBLE_DEVICES=1
  export CAFD_CONFIG=configs/cafd/manufactoria_mistral_cafd_gen_v10_mixed.yaml
  .venv/bin/python -m cafd.mistral_acquisition_observe     --root "$ROOT" --method cafd --label CAFD
) >> "logs/cafd/cafd-milestone-observe-v11-${SLURM_JOB_ID}.log" 2>&1 &
cafd_observe_pid=$!
status=0
wait "$gkd_pid" || status=$?
wait "$cafd_observe_pid" || status=$?
test "$status" -eq 0

write_stage progressive_milestone_observation
export CUDA_VISIBLE_DEVICES=0
export CAFD_CONFIG=configs/cafd/manufactoria_mistral_progressive_observation_v11.yaml
.venv/bin/python -m cafd.mistral_acquisition_observe   --root "$ROOT" --method progressive --label "Progressive GKD"   >> "logs/cafd/progressive-milestone-observe-v11-${SLURM_JOB_ID}.log" 2>&1

write_stage freeze_mechanism_observation
unset CAFD_CONFIG
.venv/bin/python -m cafd.mistral_acquisition_compare --root "$ROOT"
write_stage complete
