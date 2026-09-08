#!/usr/bin/env bash
#SBATCH --job-name=gap-bootstrap
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=02:00:00
#SBATCH --output=/blue/du.j/jinjiaguo/CAFD/logs/gap_validation/bootstrap-%j.out
#SBATCH --error=/blue/du.j/jinjiaguo/CAFD/logs/gap_validation/bootstrap-%j.err
set -euo pipefail
/blue/du.j/jinjiaguo/CAFD/scripts/gap_validation/bootstrap_env.sh
