from __future__ import annotations

import torch

from cafd.relative_target import log_odds_shift, relative_target_distribution, relative_target_logits


def test_identical_teacher_pair_recovers_phase_reference() -> None:
    torch.manual_seed(1)
    phase = torch.randn(3, 17)
    teacher = torch.randn(3, 17)
    q = relative_target_distribution(phase, teacher, teacher)
    torch.testing.assert_close(q, phase.softmax(-1), atol=2e-5, rtol=2e-5)


def test_phase_reference_equal_previous_recovers_next_teacher() -> None:
    torch.manual_seed(2)
    previous = torch.randn(5, 13)
    next_teacher = torch.randn(5, 13)
    q = relative_target_distribution(previous, next_teacher, previous)
    torch.testing.assert_close(q, next_teacher.softmax(-1), atol=2e-5, rtol=2e-5)


def test_additive_logit_constants_do_not_change_target() -> None:
    torch.manual_seed(3)
    phase, next_teacher, previous = [torch.randn(4, 19) for _ in range(3)]
    reference = relative_target_distribution(phase, next_teacher, previous)
    shifted = relative_target_distribution(phase + 8.3, next_teacher - 5.1, previous + 2.7)
    torch.testing.assert_close(shifted, reference, atol=2e-5, rtol=2e-5)


def test_teacher_log_odds_shift_is_applied_exactly() -> None:
    torch.manual_seed(4)
    phase, next_teacher, previous = [torch.randn(7, 11) for _ in range(3)]
    target = relative_target_logits(phase, next_teacher, previous)
    for left, right in [(0, 1), (3, 9), (10, 2)]:
        expected = log_odds_shift(phase, left, right) + log_odds_shift(next_teacher, left, right) - log_odds_shift(
            previous, left, right
        )
        torch.testing.assert_close(log_odds_shift(target, left, right), expected, atol=2e-5, rtol=2e-5)


def test_target_does_not_degenerate_to_progressive_gkd() -> None:
    torch.manual_seed(5)
    phase = torch.randn(2, 23)
    previous = torch.randn(2, 23)
    next_teacher = torch.randn(2, 23)
    assert not torch.allclose(phase, previous)
    q = relative_target_distribution(phase, next_teacher, previous)
    assert not torch.allclose(q, next_teacher.softmax(-1), atol=1e-5, rtol=1e-5)


def test_all_target_tensors_are_detached() -> None:
    tensors = [torch.randn(3, 7, requires_grad=True) for _ in range(3)]
    logits = relative_target_logits(*tensors)
    target = relative_target_distribution(*tensors)
    assert not logits.requires_grad
    assert not target.requires_grad
    assert logits.grad_fn is None
    assert target.grad_fn is None
