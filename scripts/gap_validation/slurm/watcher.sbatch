#!/usr/bin/env bash
#SBATCH --job-name=gap-watcher
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=14-00:00:00
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/gap_validation/watcher-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/gap_validation/watcher-%j.err
set -euo pipefail
/blue/du.j/jinjiaguo/CAFD/scripts/gap_validation/watch.sh
