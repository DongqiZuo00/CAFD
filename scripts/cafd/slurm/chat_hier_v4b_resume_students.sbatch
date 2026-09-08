#!/usr/bin/env bash
#SBATCH --job-name=cafd-v4b-resume
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/cafd/v4b-resume-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/cafd/v4b-resume-%j.err

set -euo pipefail
ROOT=/blue/du.j/jinjiaguo/CAFD
RUN_ID=chat_hier_v4b_sft25
STATE_ROOT="$ROOT/state/cafd/experiments/$RUN_ID"
PRIOR_GPU_HOURS=${CAFD_PRIOR_GPU_HOURS:-5.639166666666667}
cd "$ROOT"
mkdir -p logs/cafd "$STATE_ROOT"
exec 9>"$STATE_ROOT/formal_execution.lock"
if ! flock -n 9; then
  echo "another formal CAFD execution owns the experiment lock" >&2
  exit 75
fi
export CAFD_CONFIG=configs/cafd/manufactoria_v4b_sft25.yaml
export PYTHONPATH="$ROOT:$ROOT/vendor/trl"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$ROOT/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/hub"
export TORCH_HOME="$ROOT/.cache/torch"
export XDG_CACHE_HOME="$ROOT/.cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

write_stage() {
  export CAFD_RECOVERY_STAGE=$1
  .venv/bin/python -c 'import json,os,pathlib,time; p=pathlib.Path(os.environ["STATE_ROOT"]) / "recovery_job.json"; t=p.with_suffix(p.suffix+".tmp"); t.write_text(json.dumps({"stage":os.environ["CAFD_RECOVERY_STAGE"],"job_id":os.environ["SLURM_JOB_ID"],"time":time.time()},indent=2,sort_keys=True)+"\n",encoding="utf-8"); os.replace(t,p)'
}
failure() {
  recovery_code=${1:-1}
  trap - ERR TERM INT
  export CAFD_RECOVERY_EXIT_CODE=$recovery_code
  .venv/bin/python -c 'import json,os,pathlib,time; p=pathlib.Path(os.environ["STATE_ROOT"]) / "recovery_job.json"; t=p.with_suffix(p.suffix+".tmp"); t.write_text(json.dumps({"stage":"failed","exit_code":int(os.environ["CAFD_RECOVERY_EXIT_CODE"]),"job_id":os.environ["SLURM_JOB_ID"],"time":time.time()},indent=2,sort_keys=True)+"\n",encoding="utf-8"); os.replace(t,p)' || true
  exit "$recovery_code"
}
export STATE_ROOT
trap 'failure $?' ERR
trap 'failure 143' TERM
trap 'failure 130' INT

: "${CAFD_EXPECTED_COMMIT:?missing CAFD_EXPECTED_COMMIT}"
: "${CAFD_ENDPOINT_SELECTED_SHA:?missing CAFD_ENDPOINT_SELECTED_SHA}"
: "${CAFD_PROGRESSIVE_SELECTED_SHA:?missing CAFD_PROGRESSIVE_SELECTED_SHA}"
test "$(git rev-parse HEAD)" = "$CAFD_EXPECTED_COMMIT"
git diff --quiet
git diff --cached --quiet

write_stage recovery_preflight
.venv/bin/python -m cafd.recovery_v4b \
  --root "$ROOT" \
  --phase pre-resume \
  --endpoint-sha "$CAFD_ENDPOINT_SELECTED_SHA" \
  --progressive-sha "$CAFD_PROGRESSIVE_SELECTED_SHA" \
  --source-job-id 40852412 \
  --prior-gpu-hours "$PRIOR_GPU_HOURS"

attempt_name="raw_rollouts.resume-${SLURM_JOB_ID}.jsonl"
test ! -e "$ROOT/runs/cafd/experiments/$RUN_ID/direct/$attempt_name"
test ! -e "$ROOT/runs/cafd/experiments/$RUN_ID/cafd/$attempt_name"
write_stage student_resume

set +e
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-task=1 --mem=60G \
  .venv/bin/python -m cafd.train_student \
  --root "$ROOT" \
  --method direct \
  --rollout-log-name "$attempt_name" \
  --prior-gpu-hours "$PRIOR_GPU_HOURS" \
  >> "logs/cafd/direct-v4b-resume-${SLURM_JOB_ID}.log" 2>&1 &
direct_pid=$!
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-task=1 --mem=60G \
  .venv/bin/python -m cafd.train_student \
  --root "$ROOT" \
  --method cafd \
  --rollout-log-name "$attempt_name" \
  --prior-gpu-hours "$PRIOR_GPU_HOURS" \
  >> "logs/cafd/cafd-v4b-resume-${SLURM_JOB_ID}.log" 2>&1 &
cafd_pid=$!
wait "$direct_pid"
direct_code=$?
wait "$cafd_pid"
cafd_code=$?
set -e
if [[ "$direct_code" -ne 0 || "$cafd_code" -ne 0 ]]; then
  echo "student resume failed: direct=$direct_code cafd=$cafd_code" >&2
  recovery_code=$direct_code
  if [[ "$recovery_code" -eq 0 ]]; then
    recovery_code=$cafd_code
  fi
  failure "$recovery_code"
fi

write_stage student_matrix_audit
.venv/bin/python -m cafd.recovery_v4b \
  --root "$ROOT" \
  --phase post-students \
  --endpoint-sha "$CAFD_ENDPOINT_SELECTED_SHA" \
  --progressive-sha "$CAFD_PROGRESSIVE_SELECTED_SHA"
write_stage student_matrix_complete
