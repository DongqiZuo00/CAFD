from pathlib import Path

import yaml

from cafd import mistral_cafd_support_probe as probe


def test_v8_config_is_exactly_ten_step_next_teacher_probe() -> None:
    config = yaml.safe_load(
        Path("configs/cafd/manufactoria_mistral_cafd_support_probe_v8.yaml").read_text(
            encoding="utf-8"
        )
    )
    probe.validate_probe_config(config)
    assert config["student"]["updates"] == 10
    assert config["cafd"]["rollout_support"] == "next_teacher_replay"


def test_support_probe_generates_every_prefix_from_next_teacher(monkeypatch) -> None:
    student = object()
    teacher = object()
    seen = {}

    def fake_generate(model, *args, **kwargs):
        seen["model"] = model
        return [{"row_id": "x", "prompt_ids": [1], "completion_ids": [2]}]

    monkeypatch.setattr(probe, "_support_model", teacher)
    monkeypatch.setattr(probe, "_base_generate_rollouts", fake_generate)
    rollouts = probe._teacher_replay_rollouts(
        student,
        object(),
        [],
        [],
        0,
        {},
        2048,
        object(),
        "ministral3_instruct_fixed_system",
    )
    assert seen["model"] is teacher
    assert rollouts[0]["rollout_source"] == "R1_next_teacher_replay"
