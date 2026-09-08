from pathlib import Path

import torch
import yaml

from cafd import mistral_cafd_support_full as full


def test_full_config_is_exact_200_update_cafd_only_protocol() -> None:
    config = yaml.safe_load(
        Path("configs/cafd/manufactoria_mistral_cafd_support_full_v9.yaml").read_text(
            encoding="utf-8"
        )
    )
    full.validate_full_config(config)
    assert config["student"]["updates"] == 200
    assert config["student"]["prompts_per_update"] == 4
    assert config["student"]["rollouts_per_prompt"] == 8
    assert config["cafd"]["rollout_support"] == "next_teacher_replay"
    assert config["final"]["conditions"] == ["cafd"]


def test_each_phase_uses_its_next_frozen_teacher(monkeypatch) -> None:
    student = object()
    teachers = {f"R{index}": object() for index in range(1, 6)}
    seen = []

    def fake_generate(model, *args, **kwargs):
        seen.append(model)
        return [{"row_id": "x", "prompt_ids": [1], "completion_ids": [2]}]

    monkeypatch.setattr(full, "_support_models", teachers)
    monkeypatch.setattr(full, "_base_generate_rollouts", fake_generate)
    for phase in range(5):
        rollouts = full._teacher_replay_rollouts(
            student,
            object(),
            [],
            [],
            phase * 40,
            {},
            2048,
            object(),
            "ministral3_instruct_fixed_system",
        )
        assert seen[-1] is teachers[f"R{phase + 1}"]
        assert rollouts[0]["support_phase"] == phase
        assert rollouts[0]["rollout_source"] == f"R{phase + 1}_next_teacher_replay"


def test_no_hash_phase_controller_preserves_and_restores_reference(tmp_path) -> None:
    student = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    controller = full.NoHashPhaseController(tmp_path / "references")
    reference, state = controller.begin_phase(
        student,
        phase_index=0,
        global_update=0,
        prompt_cursor=0,
    )
    frozen_weights = {
        name: value.detach().clone() for name, value in reference.state_dict().items()
    }
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.add_(1.0)
    assert full._reference_identity(reference) == state.phase_reference_hash
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, frozen_weights[name])

    state.phase_local_update = 40
    state.global_update = 40
    state.prompt_cursor = 160
    resume = tmp_path / "resume.pt"
    controller.save_resume(
        resume,
        student=student,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        extra={"generated_tokens": 123},
    )

    restored_student = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored_student.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer, lambda _: 1.0
    )
    restored_state, restored_reference, extra = controller.load_resume(
        resume,
        student=restored_student,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        device="cpu",
    )
    assert restored_state.global_update == 40
    assert extra["generated_tokens"] == 123
    assert full._reference_identity(restored_reference) == state.phase_reference_hash
    for name, value in restored_reference.state_dict().items():
        torch.testing.assert_close(value, frozen_weights[name])
