#!/usr/bin/env bash
#SBATCH --job-name=minimal-gap-watcher
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=14-00:00:00
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/gap_validation/minimal/watcher-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/gap_validation/minimal/watcher-%j.err
set -euo pipefail
source /blue/du.j/jinjiaguo/CAFD/scripts/gap_validation/common.sh
python -m gap_validation.minimal_scheduler
