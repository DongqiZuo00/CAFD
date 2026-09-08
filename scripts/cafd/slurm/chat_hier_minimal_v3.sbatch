#!/usr/bin/env bash
#SBATCH --job-name=cafd-chat-hier-v3
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=10-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/cafd/chat-hier-v3-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/cafd/chat-hier-v3-%j.err

set -euo pipefail

ROOT=/blue/du.j/jinjiaguo/CAFD
RUN_ID=chat_hier_v3
RUN_ROOT="$ROOT/runs/cafd/experiments/$RUN_ID"
STATE_ROOT="$ROOT/state/cafd/experiments/$RUN_ID"
cd "$ROOT"
mkdir -p logs/cafd "$RUN_ROOT" "$STATE_ROOT" artifacts/cafd/experiments/"$RUN_ID" .cache/huggingface
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
  stage=$1
  .venv/bin/python -c "import json,pathlib,time; p=pathlib.Path('$STATE_ROOT/job.json'); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps({'stage':'$stage','job_id':'$SLURM_JOB_ID','time':time.time()},indent=2)+'\n')"
}

failure() {
  code=$?
  .venv/bin/python -c "import json,pathlib,time; p=pathlib.Path('$STATE_ROOT/job.json'); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps({'stage':'failed','exit_code':$code,'job_id':'$SLURM_JOB_ID','time':time.time()},indent=2)+'\n')" || true
  exit "$code"
}
trap failure ERR

requeue_before_timeout() {
  write_stage requeue_for_walltime || true
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_before_timeout USR1

.venv/bin/python -c "import json,pathlib,yaml; c=yaml.safe_load(pathlib.Path('configs/cafd/manufactoria_has.yaml').read_text()); assert c['protocol']=='compute_bounded_v3_chat_hierarchical'; assert c['experiment']['run_id']=='$RUN_ID'; g=json.loads(pathlib.Path('artifacts/cafd/renderer_gate_chat_hier_v3.json').read_text()); assert g['status']=='passed' and g['chat']['variable_reward_groups']>=1 and g['chat']['parse_or_better']>=1"
.venv/bin/python -m pytest tests/cafd -q

write_stage bootstrap
.venv/bin/python -m cafd.bootstrap --root "$ROOT" >> logs/cafd/bootstrap-chat-hier-v3.log 2>&1

write_stage capacity
.venv/bin/torchrun --standalone --nproc_per_node=4 -m cafd.train_capacity --root "$ROOT" >> logs/cafd/capacity-chat-hier-v3.log 2>&1
.venv/bin/python -c "import json,pathlib; g=json.loads(pathlib.Path('$RUN_ROOT/capacity/gate.json').read_text()); assert g['status']=='passed'"

write_stage teacher
set +e
timeout --signal=TERM --kill-after=120s 7d .venv/bin/torchrun --standalone --nproc_per_node=4 -m cafd.train_teacher --root "$ROOT" >> logs/cafd/teacher-chat-hier-v3.log 2>&1
teacher_code=$?
set -e
if [[ "$teacher_code" -ne 0 ]]; then
  if [[ "$teacher_code" -eq 124 || "$teacher_code" -eq 137 || "$teacher_code" -eq 143 ]]; then
    .venv/bin/python -c "import json,pathlib,time; p=pathlib.Path('$RUN_ROOT/teacher/gate.json'); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps({'status':'TEACHER_ROUTE_FAILED','reason':'seven_day_hard_timeout','time':time.time()},indent=2)+'\n')"
  fi
  exit "$teacher_code"
fi
.venv/bin/python -c "import json,pathlib; g=json.loads(pathlib.Path('$RUN_ROOT/teacher/gate.json').read_text()); r=json.loads(pathlib.Path('$RUN_ROOT/teacher/route.json').read_text()); assert g['status']=='passed' and r['status']=='frozen' and len(r['checkpoints'])==6"

write_stage student_matrix
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus-per-task=1 --mem=44G .venv/bin/python -m cafd.train_student --root "$ROOT" --method direct >> logs/cafd/direct-chat-hier-v3.log 2>&1 &
pid_direct=$!
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus-per-task=1 --mem=44G .venv/bin/python -m cafd.train_student --root "$ROOT" --method endpoint >> logs/cafd/endpoint-chat-hier-v3.log 2>&1 &
pid_endpoint=$!
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus-per-task=1 --mem=44G .venv/bin/python -m cafd.train_student --root "$ROOT" --method progressive >> logs/cafd/progressive-chat-hier-v3.log 2>&1 &
pid_progressive=$!
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus-per-task=1 --mem=44G .venv/bin/python -m cafd.train_student --root "$ROOT" --method cafd >> logs/cafd/cafd-chat-hier-v3.log 2>&1 &
pid_cafd=$!
wait "$pid_direct"
wait "$pid_endpoint"
wait "$pid_progressive"
wait "$pid_cafd"

write_stage freeze
.venv/bin/python -m cafd.freeze --root "$ROOT" >> logs/cafd/freeze-chat-hier-v3.log 2>&1

write_stage frozen_test
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus-per-task=1 --mem=44G .venv/bin/python -m cafd.evaluate_condition --root "$ROOT" --condition teacher_base >> logs/cafd/final-teacher_base-chat-hier-v3.log 2>&1 &
pid_e1=$!
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus-per-task=1 --mem=44G .venv/bin/python -m cafd.evaluate_condition --root "$ROOT" --condition teacher_rlvr >> logs/cafd/final-teacher_rlvr-chat-hier-v3.log 2>&1 &
pid_e2=$!
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus-per-task=1 --mem=44G .venv/bin/python -m cafd.evaluate_condition --root "$ROOT" --condition student_base >> logs/cafd/final-student_base-chat-hier-v3.log 2>&1 &
pid_e3=$!
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus-per-task=1 --mem=44G .venv/bin/python -m cafd.evaluate_condition --root "$ROOT" --condition capacity >> logs/cafd/final-capacity-chat-hier-v3.log 2>&1 &
pid_e4=$!
wait "$pid_e1"
wait "$pid_e2"
wait "$pid_e3"
wait "$pid_e4"

srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus-per-task=1 --mem=44G .venv/bin/python -m cafd.evaluate_condition --root "$ROOT" --condition direct >> logs/cafd/final-direct-chat-hier-v3.log 2>&1 &
pid_e1=$!
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus-per-task=1 --mem=44G .venv/bin/python -m cafd.evaluate_condition --root "$ROOT" --condition endpoint >> logs/cafd/final-endpoint-chat-hier-v3.log 2>&1 &
pid_e2=$!
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus-per-task=1 --mem=44G .venv/bin/python -m cafd.evaluate_condition --root "$ROOT" --condition progressive >> logs/cafd/final-progressive-chat-hier-v3.log 2>&1 &
pid_e3=$!
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus-per-task=1 --mem=44G .venv/bin/python -m cafd.evaluate_condition --root "$ROOT" --condition cafd >> logs/cafd/final-cafd-chat-hier-v3.log 2>&1 &
pid_e4=$!
wait "$pid_e1"
wait "$pid_e2"
wait "$pid_e3"
wait "$pid_e4"

.venv/bin/python -m cafd.finalize --root "$ROOT" >> logs/cafd/finalize-chat-hier-v3.log 2>&1
write_stage complete
