#!/usr/bin/env bash
set -euo pipefail
source /blue/du.j/jinjiaguo/CAFD/scripts/gap_validation/common.sh
python -m gap_validation.scheduler refresh
python -m gap_validation.scheduler dispatch
