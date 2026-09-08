#!/usr/bin/env bash
set -euo pipefail
source /blue/du.j/jinjiaguo/CAFD/scripts/gap_validation/common.sh
export NCCL_DEBUG=INFO
torchrun --standalone --nproc_per_node=4 -m gap_validation.hardware
