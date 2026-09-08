#!/usr/bin/env bash
#SBATCH --job-name=cafd-only-v5
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/cafd/cafd-only-v5-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/cafd/cafd-only-v5-%j.err

set -euo pipefail
ROOT=/blue/du.j/jinjiaguo/CAFD
RUN_ID=cafd_only_v5_full_route
STATE_ROOT="$ROOT/state/cafd/experiments/$RUN_ID"
cd "$ROOT"
mkdir -p logs/cafd "$STATE_ROOT"
exec 9>"$STATE_ROOT/formal_execution.lock"
if ! flock -n 9; then
  echo "another CAFD-only v5 execution owns the experiment lock" >&2
  exit 75
fi
export CAFD_CONFIG=configs/cafd/manufactoria_cafd_only_v5.yaml
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
  export CAFD_V5_STAGE=$1
  .venv/bin/python -c 'import json,os,pathlib,time; p=pathlib.Path(os.environ["STATE_ROOT"])/"job.json"; t=p.with_suffix(".json.tmp"); t.write_text(json.dumps({"stage":os.environ["CAFD_V5_STAGE"],"job_id":os.environ["SLURM_JOB_ID"],"time":time.time()},indent=2,sort_keys=True)+"\n",encoding="utf-8"); os.replace(t,p)'
}
failure() {
  final_code=${1:-1}
  trap - ERR TERM INT
  export CAFD_V5_EXIT_CODE=$final_code
  .venv/bin/python -c 'import json,os,pathlib,time; p=pathlib.Path(os.environ["STATE_ROOT"])/"job.json"; t=p.with_suffix(".json.tmp"); t.write_text(json.dumps({"stage":"failed","exit_code":int(os.environ["CAFD_V5_EXIT_CODE"]),"job_id":os.environ["SLURM_JOB_ID"],"time":time.time()},indent=2,sort_keys=True)+"\n",encoding="utf-8"); os.replace(t,p)' || true
  exit "$final_code"
}
trap 'failure $?' ERR
trap 'failure 143' TERM
trap 'failure 130' INT

: "${CAFD_EXPECTED_COMMIT:?missing CAFD_EXPECTED_COMMIT}"
test "$(git rev-parse HEAD)" = "$CAFD_EXPECTED_COMMIT"
git diff --quiet
git diff --cached --quiet

write_stage setup
.venv/bin/python -m cafd.setup_cafd_only_v5 --root "$ROOT"
write_stage training
.venv/bin/python -m cafd.train_student --root "$ROOT" --method cafd +  >> "logs/cafd/cafd-only-v5-train-${SLURM_JOB_ID}.log" 2>&1
write_stage complete
