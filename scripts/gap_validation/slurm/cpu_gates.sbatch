#!/usr/bin/env bash
set -euo pipefail
source /blue/du.j/jinjiaguo/CAFD/scripts/gap_validation/common.sh

mkdir -p artifacts/gap_validation/tests
pytest tests/gap_validation \
  --junitxml=artifacts/gap_validation/tests/pytest.xml \
  | tee artifacts/gap_validation/tests/pytest.log
python -m gap_validation.data all \
  | tee artifacts/gap_validation/tests/data_gate.log
python -m gap_validation.tokenizer_audit --pair qwen \
  | tee artifacts/gap_validation/tests/tokenizer_qwen.log
python -m gap_validation.model_contract \
  | tee artifacts/gap_validation/tests/model_contract.log
python -m gap_validation.config_contract \
  | tee artifacts/gap_validation/tests/config_contract.log
