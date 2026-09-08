from pathlib import Path

import yaml

from cafd.mistral_disjoint import (
    EXPECTED_TRAINING_TASKS,
    deterministic_partition,
)
from cafd.mistral_runtime import V7_RUN_ID, validate_config


ROOT = Path(__file__).resolve().parents[2]


def test_v7_config_is_pinned() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / "configs/cafd/manufactoria_mistral_cafd_disjoint_v7.yaml"
        ).read_text()
    )
    validate_config(config)
    assert config["experiment"]["run_id"] == V7_RUN_ID
    assert config["data_split"]["teacher_sft_solutions"] == 512
    assert config["data_split"]["teacher_rlvr_prompts"] == 166


def test_teacher_partition_is_reproducible_disjoint_and_complete() -> None:
    task_ids = [f"task-{index}" for index in range(EXPECTED_TRAINING_TASKS)]
    first = deterministic_partition(task_ids, seed=2027, sft_count=512)
    second = deterministic_partition(task_ids, seed=2027, sft_count=512)
    assert first == second
    sft_ids, rlvr_ids = first
    assert len(sft_ids) == 512
    assert len(rlvr_ids) == 166
    assert not set(sft_ids) & set(rlvr_ids)
    assert set(sft_ids) | set(rlvr_ids) == set(task_ids)
