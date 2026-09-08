#!/usr/bin/env bash
#SBATCH --job-name=cafd-v4b-final
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/cafd/v4b-final-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/cafd/v4b-final-%j.err

set -euo pipefail
ROOT=/blue/du.j/jinjiaguo/CAFD
RUN_ID=chat_hier_v4b_sft25
STATE_ROOT="$ROOT/state/cafd/experiments/$RUN_ID"
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
  export CAFD_FINAL_STAGE=$1
  .venv/bin/python -c 'import json,os,pathlib,time; p=pathlib.Path(os.environ["STATE_ROOT"]) / "final_job.json"; t=p.with_suffix(p.suffix+".tmp"); t.write_text(json.dumps({"stage":os.environ["CAFD_FINAL_STAGE"],"job_id":os.environ["SLURM_JOB_ID"],"dependency_job_id":os.environ["CAFD_RESUME_JOB_ID"],"time":time.time()},indent=2,sort_keys=True)+"\n",encoding="utf-8"); os.replace(t,p)'
}
failure() {
  final_code=${1:-1}
  trap - ERR TERM INT
  export CAFD_FINAL_EXIT_CODE=$final_code
  .venv/bin/python -c 'import json,os,pathlib,time; p=pathlib.Path(os.environ["STATE_ROOT"]) / "final_job.json"; t=p.with_suffix(p.suffix+".tmp"); t.write_text(json.dumps({"stage":"failed","exit_code":int(os.environ["CAFD_FINAL_EXIT_CODE"]),"job_id":os.environ["SLURM_JOB_ID"],"dependency_job_id":os.environ["CAFD_RESUME_JOB_ID"],"time":time.time()},indent=2,sort_keys=True)+"\n",encoding="utf-8"); os.replace(t,p)' || true
  exit "$final_code"
}
run_batch() {
  local condition
  local pid
  local batch_failed=0
  local pids=()
  for condition in "$@"; do
    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus-per-task=1 --mem=44G \
      .venv/bin/python -m cafd.evaluate_condition \
      --root "$ROOT" \
      --condition "$condition" \
      >> "logs/cafd/final-${condition}-v4b-${SLURM_JOB_ID}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      batch_failed=1
    fi
  done
  return "$batch_failed"
}
export STATE_ROOT
trap 'failure $?' ERR
trap 'failure 143' TERM
trap 'failure 130' INT

: "${CAFD_EXPECTED_COMMIT:?missing CAFD_EXPECTED_COMMIT}"
: "${CAFD_ENDPOINT_SELECTED_SHA:?missing CAFD_ENDPOINT_SELECTED_SHA}"
: "${CAFD_PROGRESSIVE_SELECTED_SHA:?missing CAFD_PROGRESSIVE_SELECTED_SHA}"
: "${CAFD_RESUME_JOB_ID:?missing CAFD_RESUME_JOB_ID}"
test "$(git rev-parse HEAD)" = "$CAFD_EXPECTED_COMMIT"
git diff --quiet
git diff --cached --quiet

.venv/bin/python -c 'import json,os,pathlib; r=json.loads((pathlib.Path(os.environ["STATE_ROOT"]) / "recovery_job.json").read_text()); assert r["stage"]=="student_matrix_complete" and r["job_id"]==os.environ["CAFD_RESUME_JOB_ID"]'

write_stage pre_freeze_audit
.venv/bin/python -m cafd.recovery_v4b \
  --root "$ROOT" \
  --phase post-students \
  --endpoint-sha "$CAFD_ENDPOINT_SELECTED_SHA" \
  --progressive-sha "$CAFD_PROGRESSIVE_SELECTED_SHA"

if [[ ! -f "$ROOT/runs/cafd/experiments/$RUN_ID/frozen_selection.json" ]]; then
  write_stage freeze
  .venv/bin/python -m cafd.freeze --root "$ROOT" \
    >> "logs/cafd/freeze-v4b-${SLURM_JOB_ID}.log" 2>&1
fi
write_stage post_freeze_audit
.venv/bin/python -m cafd.recovery_v4b \
  --root "$ROOT" \
  --phase post-freeze \
  --endpoint-sha "$CAFD_ENDPOINT_SELECTED_SHA" \
  --progressive-sha "$CAFD_PROGRESSIVE_SELECTED_SHA"

write_stage frozen_test_batch_1
run_batch teacher_base teacher_rlvr student_base capacity
write_stage frozen_test_batch_2
run_batch direct endpoint progressive cafd

write_stage finalize
.venv/bin/python -m cafd.finalize --root "$ROOT" \
  >> "logs/cafd/finalize-v4b-${SLURM_JOB_ID}.log" 2>&1
write_stage final_audit
.venv/bin/python -m cafd.recovery_v4b \
  --root "$ROOT" \
  --phase post-final \
  --endpoint-sha "$CAFD_ENDPOINT_SELECTED_SHA" \
  --progressive-sha "$CAFD_PROGRESSIVE_SELECTED_SHA"
write_stage complete
