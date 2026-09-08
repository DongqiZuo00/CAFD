#!/usr/bin/env bash
#SBATCH --job-name=cafd-v4-sft25
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=10-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/cafd/chat-hier-v4-sft25-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/cafd/chat-hier-v4-sft25-%j.err

set -euo pipefail
ROOT=/blue/du.j/jinjiaguo/CAFD
RUN_ID=chat_hier_v4_sft25
RUN_ROOT="$ROOT/runs/cafd/experiments/$RUN_ID"
STATE_ROOT="$ROOT/state/cafd/experiments/$RUN_ID"
cd "$ROOT"
mkdir -p logs/cafd "$RUN_ROOT" "$STATE_ROOT" artifacts/cafd/experiments/"$RUN_ID"
export CAFD_CONFIG=configs/cafd/manufactoria_v4_sft25.yaml
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

test -d runs/cafd/experiments/teacher_capacity_v1/teacher_capacity/SFT25
test "$(.venv/bin/python -c "import json; print(json.load(open('runs/cafd/experiments/teacher_capacity_v1/teacher_capacity/gate.json'))['status'])")" = passed
if [[ ! -e "$RUN_ROOT/student_base" ]]; then ln -s ../chat_hier_v3/student_base "$RUN_ROOT/student_base"; fi
if [[ ! -e "$RUN_ROOT/capacity" ]]; then ln -s ../chat_hier_v3/capacity "$RUN_ROOT/capacity"; fi
test "$(readlink "$RUN_ROOT/student_base")" = ../chat_hier_v3/student_base
test "$(readlink "$RUN_ROOT/capacity")" = ../chat_hier_v3/capacity
.venv/bin/python -c "import pathlib,yaml; c=yaml.safe_load(pathlib.Path('$CAFD_CONFIG').read_text()); assert c['protocol']=='compute_bounded_v4_sft25_warmstart'; assert c['experiment']['run_id']=='$RUN_ID'; assert c['teacher_route']['minimum_updates']==100; assert c['generation']['prompt_renderer']=='qwen3_instruct_no_thinking'"
.venv/bin/python -m pytest tests/cafd -q

write_stage warmstart_gate
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-task=1 --mem=48G .venv/bin/python -m cafd.teacher_warmstart_gate --root "$ROOT" >> logs/cafd/warmstart-gate-v4.log 2>&1
.venv/bin/python -c "import json,pathlib; g=json.loads(pathlib.Path('artifacts/cafd/experiments/$RUN_ID/warmstart_gate.json').read_text()); assert g['status']=='passed' and g['development']['correct']==47"

write_stage teacher
set +e
timeout --signal=TERM --kill-after=120s 7d .venv/bin/torchrun --standalone --nproc_per_node=4 -m cafd.train_teacher --root "$ROOT" >> logs/cafd/teacher-chat-hier-v4-sft25.log 2>&1
teacher_code=$?
set -e
if [[ "$teacher_code" -ne 0 ]]; then exit "$teacher_code"; fi
.venv/bin/python -c "import json,pathlib; g=json.loads(pathlib.Path('$RUN_ROOT/teacher/gate.json').read_text()); r=json.loads(pathlib.Path('$RUN_ROOT/teacher/route.json').read_text()); assert g['status']=='passed' and g['actual_updates']>=100 and r['status']=='frozen' and len(r['checkpoints'])==6"

write_stage student_matrix
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus-per-task=1 --mem=44G .venv/bin/python -m cafd.train_student --root "$ROOT" --method direct >> logs/cafd/direct-chat-hier-v4-sft25.log 2>&1 &
p1=$!
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus-per-task=1 --mem=44G .venv/bin/python -m cafd.train_student --root "$ROOT" --method endpoint >> logs/cafd/endpoint-chat-hier-v4-sft25.log 2>&1 &
p2=$!
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus-per-task=1 --mem=44G .venv/bin/python -m cafd.train_student --root "$ROOT" --method progressive >> logs/cafd/progressive-chat-hier-v4-sft25.log 2>&1 &
p3=$!
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus-per-task=1 --mem=44G .venv/bin/python -m cafd.train_student --root "$ROOT" --method cafd >> logs/cafd/cafd-chat-hier-v4-sft25.log 2>&1 &
p4=$!
wait "$p1"; wait "$p2"; wait "$p3"; wait "$p4"

write_stage freeze
.venv/bin/python -m cafd.freeze --root "$ROOT" >> logs/cafd/freeze-chat-hier-v4-sft25.log 2>&1

write_stage frozen_test
for condition in teacher_base teacher_rlvr student_base capacity; do
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus-per-task=1 --mem=44G .venv/bin/python -m cafd.evaluate_condition --root "$ROOT" --condition "$condition" >> "logs/cafd/final-$condition-chat-hier-v4-sft25.log" 2>&1 &
done
wait
for condition in direct endpoint progressive cafd; do
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus-per-task=1 --mem=44G .venv/bin/python -m cafd.evaluate_condition --root "$ROOT" --condition "$condition" >> "logs/cafd/final-$condition-chat-hier-v4-sft25.log" 2>&1 &
done
wait
.venv/bin/python -m cafd.finalize --root "$ROOT" >> logs/cafd/finalize-chat-hier-v4-sft25.log 2>&1
write_stage complete
