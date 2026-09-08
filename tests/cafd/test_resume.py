from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from cafd.exact_forward_kl import projected_relative_forward_kl
from cafd.phase_controller import PhaseController, parameter_sha256
from cafd.training_common import checkpoint_complete
from cafd.train_student import (
    _curve_milestone_count,
    _bootstrap_gate_payload,
    _physical_rollout_tokens,
    _raw_rollout_path,
    _validated_prior_gpu_hours,
)


class ToyStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(5, 7)
        self.head = nn.Linear(7, 11, bias=False)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.backbone(x))


def _loss(student: ToyStudent, phase: ToyStudent, previous: ToyStudent, next_teacher: ToyStudent, x: torch.Tensor) -> torch.Tensor:
    mask = torch.ones(x.shape[:2], dtype=torch.bool)
    return projected_relative_forward_kl(
        student.hidden(x),
        phase.hidden(x),
        next_teacher.hidden(x),
        previous.hidden(x),
        student.head,
        phase.head,
        next_teacher.head,
        previous.head,
        mask,
        token_block_size=3,
    )


def test_phase_reference_is_immutable_and_storage_independent(tmp_path) -> None:
    torch.manual_seed(21)
    student = ToyStudent()
    controller = PhaseController(tmp_path / "phases")
    phase, state = controller.begin_phase(student, phase_index=0, global_update=0, prompt_cursor=0)
    initial_hash = parameter_sha256(phase)
    for source, target in zip(student.parameters(), phase.parameters(), strict=True):
        assert source.data_ptr() != target.data_ptr()
        assert not target.requires_grad
    with torch.no_grad():
        next(student.parameters()).add_(3.0)
    assert parameter_sha256(phase) == initial_hash == state.phase_reference_hash


def test_mid_phase_resume_preserves_target_loss_and_next_update(tmp_path) -> None:
    torch.manual_seed(22)
    student = ToyStudent()
    previous = ToyStudent().eval().requires_grad_(False)
    next_teacher = ToyStudent().eval().requires_grad_(False)
    controller = PhaseController(tmp_path / "phases")
    phase, state = controller.begin_phase(student, phase_index=2, global_update=91, prompt_cursor=728)
    state.phase_local_update = 11
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    x = torch.randn(2, 4, 5)

    # Establish non-empty optimizer state, then checkpoint mid-phase.
    first = _loss(student, phase, previous, next_teacher, x)
    first.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state.global_update += 1
    state.phase_local_update += 1
    resume_path = tmp_path / "resume.pt"
    controller.save_resume(
        resume_path,
        student=student,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        extra={"prompt_ids": [1, 2, 3]},
    )

    uninterrupted_student = copy.deepcopy(student)
    uninterrupted_optimizer = torch.optim.AdamW(uninterrupted_student.parameters(), lr=1e-3)
    uninterrupted_optimizer.load_state_dict(optimizer.state_dict())
    uninterrupted_phase = copy.deepcopy(phase)
    target_loss = _loss(uninterrupted_student, uninterrupted_phase, previous, next_teacher, x)
    target_loss.backward()
    uninterrupted_optimizer.step()

    resumed_student = ToyStudent()
    resumed_optimizer = torch.optim.AdamW(resumed_student.parameters(), lr=1e-3)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda _: 1.0)
    restored_state, restored_phase, extra = controller.load_resume(
        resume_path,
        student=resumed_student,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        device="cpu",
    )
    resumed_loss = _loss(resumed_student, restored_phase, previous, next_teacher, x)
    resumed_loss.backward()
    resumed_optimizer.step()

    torch.testing.assert_close(resumed_loss, target_loss, atol=2e-5, rtol=2e-5)
    for actual, expected in zip(resumed_student.parameters(), uninterrupted_student.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)
    assert parameter_sha256(restored_phase) == state.phase_reference_hash
    assert restored_state.phase_local_update == state.phase_local_update
    assert restored_state.prompt_cursor == state.prompt_cursor
    assert extra == {"prompt_ids": [1, 2, 3]}


def test_curve_milestone_count_is_exact(tmp_path) -> None:
    curve = tmp_path / "curve.csv"
    curve.write_text(
        "condition,step\nDirect RLVR,80\nDirect RLVR,120\nCAFD-v1,120\n",
        encoding="utf-8",
    )
    assert _curve_milestone_count(curve, "Direct RLVR", 120) == 1
    assert _curve_milestone_count(curve, "CAFD-v1", 120) == 1
    assert _curve_milestone_count(curve, "Direct RLVR", 160) == 0
    assert _curve_milestone_count(tmp_path / "missing.csv", "Direct RLVR", 120) == 0


def test_recovery_rollout_log_must_be_jsonl_basename(tmp_path) -> None:
    assert _raw_rollout_path(tmp_path, "attempt.jsonl") == tmp_path / "attempt.jsonl"
    for invalid in ("../attempt.jsonl", "nested/attempt.jsonl", "attempt.csv", ""):
        with pytest.raises(ValueError, match="jsonl basename"):
            _raw_rollout_path(tmp_path, invalid)


def test_prior_gpu_hours_validation() -> None:
    assert _validated_prior_gpu_hours(5.5) == 5.5
    assert _validated_prior_gpu_hours(0) == 0.0
    for invalid in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and non-negative"):
            _validated_prior_gpu_hours(invalid)


def test_physical_rollout_tokens_aggregates_all_attempts(tmp_path) -> None:
    (tmp_path / "raw_rollouts.jsonl").write_text(
        '{"completion_ids":[1,2]}\n',
        encoding="utf-8",
    )
    (tmp_path / "raw_rollouts.resume-123.jsonl").write_text(
        '{"completion_ids":[3]}\n{"completion_ids":[4,5,6]}\n',
        encoding="utf-8",
    )
    assert _physical_rollout_tokens(tmp_path) == 6


def test_corrupt_phase_reference_is_quarantined_and_rebuilt(tmp_path) -> None:
    torch.manual_seed(31)
    student = ToyStudent()
    controller = PhaseController(tmp_path / "phases")
    target = controller.root / "phase-00-reference.pt"
    target.write_bytes(b"truncated")

    phase, state = controller.begin_phase(
        student,
        phase_index=0,
        global_update=0,
        prompt_cursor=0,
    )

    payload = torch.load(target, map_location="cpu", weights_only=False)
    assert payload["sha256"] == state.phase_reference_hash
    assert parameter_sha256(phase) == state.phase_reference_hash
    assert len(list(controller.root.glob("phase-00-reference.pt.aborted-*"))) == 1


def test_checkpoint_complete_rejects_missing_payload(tmp_path) -> None:
    checkpoint = tmp_path / "step"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"0" * 1_000_001)

    assert checkpoint_complete(checkpoint)
    weights.unlink()
    assert not checkpoint_complete(checkpoint)


def test_cafd_bootstrap_gate_requires_real_contract_support() -> None:
    criteria = {
        "update": 10,
        "minimum_contract_valid": 1,
        "minimum_parse_or_better": 1,
        "minimum_eos_terminated": 1,
        "maximum_token_limit_rate": 0.95,
        "minimum_loss": 1e-8,
        "minimum_gradient_norm": 1e-8,
    }
    score = {
        "total": 64,
        "contract_valid": 2,
        "parse_or_better": 1,
        "eos_terminated": 2,
        "token_limit_hits": 60,
    }
    payload = _bootstrap_gate_payload(
        criteria,
        step=10,
        score=score,
        loss=0.01,
        gradient_norm=0.1,
    )
    assert payload["status"] == "passed"

    score["contract_valid"] = 0
    failed = _bootstrap_gate_payload(
        criteria,
        step=10,
        score=score,
        loss=0.01,
        gradient_norm=0.1,
    )
    assert failed["status"] == "CAFD_BOOTSTRAP_GATE_FAILED"
