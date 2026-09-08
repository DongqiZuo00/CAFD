#!/usr/bin/env bash
#SBATCH --job-name=cafd-teacher-cap-v1
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/cafd/teacher-cap-v1-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/cafd/teacher-cap-v1-%j.err

set -euo pipefail

ROOT=/blue/du.j/jinjiaguo/CAFD
RUN_ID=teacher_capacity_v1
RUN_ROOT="$ROOT/runs/cafd/experiments/$RUN_ID"
STATE_ROOT="$ROOT/state/cafd/experiments/$RUN_ID"
cd "$ROOT"
mkdir -p logs/cafd "$RUN_ROOT" "$STATE_ROOT" artifacts/cafd/experiments/"$RUN_ID"
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

.venv/bin/python -c "import json,pathlib,yaml; c=yaml.safe_load(pathlib.Path('configs/cafd/manufactoria_teacher_capacity_v1.yaml').read_text()); assert c['protocol']=='teacher_capacity_v1_chat_verified_sft'; assert c['experiment']['run_id']=='$RUN_ID'; assert c['generation']['prompt_renderer']=='qwen3_instruct_no_thinking'; p=pathlib.Path(c['teacher_capacity']['source_verified_solutions']); assert sum(1 for _ in p.open())==678; g=json.loads(pathlib.Path('artifacts/cafd/renderer_gate_chat_hier_v3.json').read_text()); assert g['status']=='passed'"
.venv/bin/python -m pytest tests/cafd -q

write_stage teacher_capacity
.venv/bin/torchrun --standalone --nproc_per_node=4 -m cafd.train_teacher_capacity --root "$ROOT" >> logs/cafd/teacher-cap-v1.log 2>&1
.venv/bin/python -c "import json,pathlib; g=json.loads(pathlib.Path('$RUN_ROOT/teacher_capacity/gate.json').read_text()); assert g['status']=='passed' and g['development_correct']>=52"
write_stage complete
