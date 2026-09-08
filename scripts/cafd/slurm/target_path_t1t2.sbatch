#!/usr/bin/env bash
#SBATCH --job-name=cafd-target-path-t1t2
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/cafd/target-path-t1t2-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/cafd/target-path-t1t2-%j.err
set -euo pipefail
cd /blue/du.j/jinjiaguo/CAFD
export PYTHONPATH="$PWD:$PWD/vendor/trl"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
run="$PWD/runs/cafd/experiments/mistral_target_path_t1t2_v1"
exec 9>"$run/execution.lock"
flock -n 9 || exit 75
.venv/bin/python -u -m cafd.target_path_resume --run-dir "$run" --job-id "$SLURM_JOB_ID" --lock-fd 9
IFS=',' read -r -a allocated_devices <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#allocated_devices[@]}" -ne 2 ]]; then
  printf '%s\n' "Expected exactly two allocated B200 devices" >&2
  exit 76
fi
CUDA_VISIBLE_DEVICES="${allocated_devices[0]}" .venv/bin/python -u -m cafd.target_path_run --worker 0 > "$run/worker0.log" 2> "$run/worker0.err" &
p0=$!
CUDA_VISIBLE_DEVICES="${allocated_devices[1]}" .venv/bin/python -u -m cafd.target_path_run --worker 1 > "$run/worker1.log" 2> "$run/worker1.err" &
p1=$!
exit0=0
exit1=0
wait "$p0" || exit0=$?
wait "$p1" || exit1=$?
printf '{"worker0_exit":%d,"worker1_exit":%d,"job_id":"%s"}\n' "$exit0" "$exit1" "$SLURM_JOB_ID" > "$run/worker_exit.json"
if [[ "$exit0" -ne 0 || "$exit1" -ne 0 ]]; then exit 1; fi
