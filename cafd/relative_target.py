"""Exact CAFD-v1 ordered-local-shift targets.

All target construction is deliberately performed in FP32 and detached.  No
ratio clipping, top-k filtering, or vocabulary remapping is allowed here.
"""

from __future__ import annotations

import torch


def _same_shape(*tensors: torch.Tensor) -> None:
    if not tensors:
        raise ValueError("at least one tensor is required")
    shape = tensors[0].shape
    if any(t.shape != shape for t in tensors[1:]):
        raise ValueError(f"all logit tensors must have the same shape, got {[tuple(t.shape) for t in tensors]}")


def relative_target_logits(
    phase_reference_logits: torch.Tensor,
    teacher_next_logits: torch.Tensor,
    teacher_previous_logits: torch.Tensor,
    *,
    gamma: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return detached FP32 logits for the ordered local-shift target.

    ``u = (a + gamma * (b - c)) / temperature`` where ``a`` is the frozen
    phase-start Student, and ``b``/``c`` are the next/previous Teacher route
    checkpoints on the exact same causal prefixes.
    """

    _same_shape(phase_reference_logits, teacher_next_logits, teacher_previous_logits)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not torch.isfinite(torch.tensor(float(gamma))):
        raise ValueError("gamma must be finite")
    with torch.no_grad():
        a = phase_reference_logits.detach().float()
        b = teacher_next_logits.detach().float()
        c = teacher_previous_logits.detach().float()
        target = (a + float(gamma) * (b - c)) / float(temperature)
    return target.detach()


def relative_target_distribution(
    phase_reference_logits: torch.Tensor,
    teacher_next_logits: torch.Tensor,
    teacher_previous_logits: torch.Tensor,
    *,
    gamma: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return the detached, full-vocabulary CAFD target distribution."""

    target_logits = relative_target_logits(
        phase_reference_logits,
        teacher_next_logits,
        teacher_previous_logits,
        gamma=gamma,
        temperature=temperature,
    )
    return torch.softmax(target_logits, dim=-1).detach()


def log_odds_shift(logits: torch.Tensor, left: int, right: int) -> torch.Tensor:
    """Convenience used by numerical tests: log p(left) - log p(right)."""

    return logits[..., left].float() - logits[..., right].float()
