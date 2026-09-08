#!/usr/bin/env bash
#SBATCH --job-name=cafd-cb
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=14-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/cafd/slurm-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/cafd/slurm-%j.err

set -euo pipefail

ROOT=/blue/du.j/jinjiaguo/CAFD
cd "$ROOT"
mkdir -p logs/cafd state/cafd runs/cafd artifacts/cafd .cache/huggingface
export PYTHONPATH="$ROOT:$ROOT/vendor/trl${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$ROOT/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/hub"
export TORCH_HOME="$ROOT/.cache/torch"
export XDG_CACHE_HOME="$ROOT/.cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

failure() {
  code=$?
  .venv/bin/python -c "import json,pathlib,time; p=pathlib.Path('state/cafd/job.json'); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps({'stage':'failed','exit_code':$code,'job_id':'${SLURM_JOB_ID:-unknown}','time':time.time()},indent=2)+'\\n')" || true
  exit "$code"
}
trap failure ERR

requeue_before_timeout() {
  .venv/bin/python -c "import json,pathlib,time; p=pathlib.Path('state/cafd/job.json'); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps({'stage':'requeue_for_walltime','job_id':'${SLURM_JOB_ID:-unknown}','time':time.time()},indent=2)+'\\n')" || true
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_before_timeout USR1

.venv/bin/python -c "import json,pathlib; p=pathlib.Path('archive/cafd_invalid_48x16x8192/BUDGET_INVALID.json'); d=json.loads(p.read_text()); assert d['status']=='BUDGET_INVALID' and d['resume_forbidden'] is True"
.venv/bin/python -m cafd.bootstrap --root "$ROOT" >> logs/cafd/bootstrap.log 2>&1
.venv/bin/torchrun --standalone --nproc_per_node=4 -m cafd.train_capacity --root "$ROOT" >> logs/cafd/capacity.log 2>&1

set +e
timeout --signal=TERM --kill-after=120s 7d \
  .venv/bin/torchrun --standalone --nproc_per_node=4 -m cafd.train_teacher --root "$ROOT" \
  >> logs/cafd/teacher.log 2>&1
teacher_code=$?
set -e
if [[ "$teacher_code" -ne 0 ]]; then
  if [[ "$teacher_code" -eq 124 || "$teacher_code" -eq 137 || "$teacher_code" -eq 143 ]]; then
    .venv/bin/python -c "import json,pathlib,time; p=pathlib.Path('runs/cafd/teacher/gate.json'); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps({'status':'TEACHER_ROUTE_FAILED','reason':'seven_day_hard_timeout','time':time.time()},indent=2)+'\n')"
  fi
  exit "$teacher_code"
fi

run_method() {
  method=$1
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus-per-task=1 --mem=44G \
    .venv/bin/python -m cafd.train_student --root "$ROOT" --method "$method" \
    >> "logs/cafd/${method}.log" 2>&1 &
  method_pids+=("$!")
}

method_pids=()
run_method direct
run_method endpoint
run_method progressive
run_method cafd
for pid in "${method_pids[@]}"; do
  wait "$pid"
done

.venv/bin/python -m cafd.freeze --root "$ROOT" >> logs/cafd/freeze.log 2>&1

run_eval() {
  condition=$1
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus-per-task=1 --mem=44G \
    .venv/bin/python -m cafd.evaluate_condition --root "$ROOT" --condition "$condition" \
    >> "logs/cafd/final-${condition}.log" 2>&1 &
  eval_pids+=("$!")
}

eval_pids=()
for condition in teacher_base teacher_rlvr student_base capacity; do
  run_eval "$condition"
done
for pid in "${eval_pids[@]}"; do
  wait "$pid"
done
eval_pids=()
for condition in direct endpoint progressive cafd; do
  run_eval "$condition"
done
for pid in "${eval_pids[@]}"; do
  wait "$pid"
done

.venv/bin/python -m cafd.finalize --root "$ROOT" >> logs/cafd/finalize.log 2>&1
.venv/bin/python -c "import json,pathlib,time; p=pathlib.Path('state/cafd/job.json'); p.write_text(json.dumps({'stage':'complete','job_id':'${SLURM_JOB_ID}','time':time.time()},indent=2)+'\\n')"
