#!/usr/bin/env bash
#SBATCH --job-name=cafd-mistral-route-v6
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=8-00:00:00
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/cafd/mistral-route-v6-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/cafd/mistral-route-v6-%j.err

set -euo pipefail
ROOT=/blue/du.j/jinjiaguo/CAFD
RUN_ID=mistral_cafd_only_v6
STATE_ROOT="$ROOT/state/cafd/experiments/$RUN_ID"
cd "$ROOT"
mkdir -p logs/cafd "$STATE_ROOT" "artifacts/cafd/experiments/$RUN_ID"
exec 9>"$STATE_ROOT/route_execution.lock"
if ! flock -n 9; then
  echo "another Mistral route execution owns the lock" >&2
  exit 75
fi
export CAFD_CONFIG=configs/cafd/manufactoria_mistral_cafd_only_v6.yaml
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
  .venv/bin/python -c 'import json,os,pathlib,time; p=pathlib.Path(os.environ["STATE_ROOT"])/"job.json"; t=p.with_suffix(".json.tmp"); t.write_text(json.dumps({"stage":os.environ["CAFD_MISTRAL_STAGE"],"job_id":os.environ["SLURM_JOB_ID"],"time":time.time()},indent=2,sort_keys=True)+"\n",encoding="utf-8"); os.replace(t,p)'
}
failure() {
  final_code=${1:-1}
  trap - ERR TERM INT
  export CAFD_MISTRAL_EXIT_CODE=$final_code
  .venv/bin/python -c 'import json,os,pathlib,time; p=pathlib.Path(os.environ["STATE_ROOT"])/"job.json"; t=p.with_suffix(".json.tmp"); t.write_text(json.dumps({"stage":"failed","exit_code":int(os.environ["CAFD_MISTRAL_EXIT_CODE"]),"job_id":os.environ["SLURM_JOB_ID"],"time":time.time()},indent=2,sort_keys=True)+"\n",encoding="utf-8"); os.replace(t,p)' || true
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

write_stage tests
.venv/bin/python -m pytest tests/cafd -q

write_stage preflight
.venv/bin/python -m cafd.mistral_preflight --root "$ROOT"

write_stage student_capacity
.venv/bin/torchrun --standalone --nproc_per_node=4 -m cafd.mistral_train_capacity --root "$ROOT" >> "logs/cafd/mistral-student-capacity-v6-${SLURM_JOB_ID}.log" 2>&1
.venv/bin/python -c 'import json; g=json.load(open("runs/cafd/experiments/mistral_cafd_only_v6/capacity/gate.json")); assert g["status"]=="passed" and g["development_correct"]>=52'

write_stage teacher_capacity
.venv/bin/torchrun --standalone --nproc_per_node=4 -m cafd.mistral_train_teacher_capacity --root "$ROOT" --config "$CAFD_CONFIG" >> "logs/cafd/mistral-teacher-capacity-v6-${SLURM_JOB_ID}.log" 2>&1
.venv/bin/python -m cafd.mistral_prepare_teacher --root "$ROOT" --mode select-seed

write_stage warmstart_gate
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-task=1 --mem=48G .venv/bin/python -m cafd.mistral_warmstart_gate --root "$ROOT" >> "logs/cafd/mistral-warmstart-v6-${SLURM_JOB_ID}.log" 2>&1

write_stage teacher_rlvr
timeout --signal=TERM --kill-after=120s 7d .venv/bin/torchrun --standalone --nproc_per_node=4 -m cafd.mistral_train_teacher --root "$ROOT" >> "logs/cafd/mistral-teacher-rlvr-v6-${SLURM_JOB_ID}.log" 2>&1
.venv/bin/python -m cafd.mistral_prepare_teacher --root "$ROOT" --mode freeze-route
write_stage route_complete
