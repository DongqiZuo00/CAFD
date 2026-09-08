from collections import Counter
from pathlib import Path

import yaml

from cafd import mistral_acquisition_observe as observe
from cafd import mistral_progressive_observation_v11 as progressive


CONFIG = Path("configs/cafd/manufactoria_mistral_progressive_observation_v11.yaml")


def test_progressive_observation_config_is_matched_and_test_sealed() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    progressive.validate_config(config)
    assert config["student"]["updates"] == 200
    assert config["student"]["prompts_per_update"] == 4
    assert config["student"]["rollouts_per_prompt"] == 8
    assert config["cafd"]["teacher_rollouts_per_prompt_by_phase"] == [6, 4, 4, 2, 2]
    assert config["cafd"]["target"] == "absolute_next_teacher_checkpoint"
    assert config["final"]["official_held_out_evaluations"] == 0


def test_progressive_support_matches_cafd_slot_schedule(monkeypatch) -> None:
    student = object()
    teachers = {f"R{index}": object() for index in range(1, 6)}
    monkeypatch.setattr(progressive, "_support_models", teachers)
    calls = []

    def fake_generate(model, tokenizer, rows, indices, update, settings, *rest):
        calls.append((model, int(settings["rollouts_per_prompt"])))
        result = []
        for prompt_slot, _ in enumerate(indices):
            for sample_slot in range(int(settings["rollouts_per_prompt"])):
                result.append({
                    "row_id": str(prompt_slot),
                    "prompt_ids": [1],
                    "completion_ids": [2],
                    "prompt_slot": prompt_slot,
                    "sample_slot": sample_slot,
                })
        return result

    monkeypatch.setattr(progressive, "_base_generate_rollouts", fake_generate)
    for phase, teacher_count in enumerate([6, 4, 4, 2, 2]):
        calls.clear()
        rollouts = progressive._support_rollouts(
            student, object(), [{}, {}], [0, 1], phase * 40,
            {"rollouts_per_prompt": 8}, 2048, object(), "renderer",
        )
        assert calls == [
            (teachers[f"R{phase + 1}"], teacher_count),
            (student, 8 - teacher_count),
        ]
        assert len(rollouts) == 16
        assert Counter(item["prompt_slot"] for item in rollouts) == {0: 8, 1: 8}


def test_observation_auc_uses_update_axis() -> None:
    records = [
        {"step": 0, "accuracy": 0.0},
        {"step": 40, "accuracy": 0.5},
        {"step": 200, "accuracy": 0.5},
    ]
    assert observe._auc(records) == 0.45
