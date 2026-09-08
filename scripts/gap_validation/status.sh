#!/usr/bin/env bash
set -euo pipefail
source /blue/du.j/jinjiaguo/CAFD/scripts/gap_validation/common.sh
python -m gap_validation.status
echo
squeue -u "${USER}" -o '%.18i %.12P %.32j %.2t %.10M %.4D %b %R'
