"""Memory-bounded exact full-vocabulary forward KL for CAFD-v1.

The implementation chunks completion *positions*, never the probability
normalization.  Every block covers the complete vocabulary and all target
arithmetic/normalization is FP32.  The Student log-normalizer remains attached,
so its exact gradient is ``(p-q)/N``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .relative_target import relative_target_logits


def completion_prediction_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
    *,
    eos_token_id: int | None,
) -> torch.Tensor:
    """Build the causal-shift mask over ``logits[:, :-1]`` predictions.

    Inputs are right padded. ``prompt_lengths`` counts non-padding prompt
    tokens. The first completion token (target index ``prompt_length``) and the
    first generated EOS are included. Prompt targets, padding, and all tokens
    after EOS are excluded. Empty completions produce an all-false row.
    """

    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must be rank-2 tensors of identical shape")
    if prompt_lengths.ndim != 1 or prompt_lengths.numel() != input_ids.shape[0]:
        raise ValueError("prompt_lengths must contain one entry per batch row")
    if input_ids.shape[1] < 2:
        return torch.zeros((input_ids.shape[0], 0), dtype=torch.bool, device=input_ids.device)

    batch, length = input_ids.shape
    target_indices = torch.arange(1, length, device=input_ids.device).expand(batch, -1)
    prompt_lengths = prompt_lengths.to(device=input_ids.device, dtype=torch.long).unsqueeze(1)
    valid = attention_mask[:, 1:].bool() & (target_indices >= prompt_lengths)

    if eos_token_id is not None:
        target_tokens = input_ids[:, 1:]
        eos_hits = (target_tokens == int(eos_token_id)) & valid
        # cumsum==0 retains positions before EOS; eos_hits itself retains EOS.
        before_or_at_first_eos = (eos_hits.cumsum(dim=1) == 0) | eos_hits & (eos_hits.cumsum(dim=1) == 1)
        valid &= before_or_at_first_eos
    return valid


def _validate_logits(student_logits: torch.Tensor, target_logits: torch.Tensor, mask: torch.Tensor) -> None:
    if student_logits.shape != target_logits.shape:
        raise ValueError("student and target logits must have identical shape")
    if student_logits.ndim != 3:
        raise ValueError("logits must have shape [batch, positions, vocabulary]")
    if mask.shape != student_logits.shape[:2]:
        raise ValueError("mask must have shape [batch, positions]")


def dense_forward_kl(
    student_logits: torch.Tensor,
    target_logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Dense FP32 reference forward KL over all valid positions."""

    _validate_logits(student_logits, target_logits, mask)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    selected_student = student_logits[mask].float() / float(temperature)
    selected_target = target_logits[mask].detach().float() / float(temperature)
    if selected_student.shape[0] == 0:
        return student_logits.sum() * 0.0
    log_p = F.log_softmax(selected_student, dim=-1)
    log_q = F.log_softmax(selected_target, dim=-1)
    q = log_q.exp()
    per_position = (q * (log_q - log_p)).sum(dim=-1)
    if reduction == "sum":
        return per_position.sum()
    if reduction == "mean":
        return per_position.mean()
    if reduction == "none":
        return per_position
    raise ValueError(f"unsupported reduction: {reduction}")


def block_forward_kl(
    student_logits: torch.Tensor,
    target_logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    token_block_size: int = 128,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Exact KL chunked by valid completion positions, with full vocabulary."""

    _validate_logits(student_logits, target_logits, mask)
    if token_block_size <= 0:
        raise ValueError("token_block_size must be positive")
    student = student_logits[mask]
    target = target_logits[mask]
    n = student.shape[0]
    if n == 0:
        return student_logits.sum() * 0.0
    total = student_logits.new_zeros((), dtype=torch.float32)
    for start in range(0, n, token_block_size):
        stop = min(start + token_block_size, n)
        total = total + dense_forward_kl(
            student[start:stop].unsqueeze(0),
            target[start:stop].unsqueeze(0),
            torch.ones((1, stop - start), dtype=torch.bool, device=mask.device),
            temperature=temperature,
            reduction="sum",
        )
    return total / n


def _linear_parts_fp32(head: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Materialize one reusable FP32 view of a complete LM head."""

    if not isinstance(head, nn.Linear):
        # Qwen heads are Linear; fail instead of silently changing semantics.
        raise TypeError(f"exact projection requires nn.Linear LM head, got {type(head).__name__}")
    bias = None if head.bias is None else head.bias.float()
    return head.weight.float(), bias


def _linear_fp32(
    hidden: torch.Tensor,
    head: nn.Module | None = None,
    *,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project a hidden-state block to the complete vocabulary in FP32."""

    if weight is None:
        if head is None:
            raise ValueError("either head or pre-materialized FP32 weight is required")
        weight, bias = _linear_parts_fp32(head)
    return F.linear(hidden.float(), weight, bias)


@dataclass(frozen=True)
class KLBlock:
    """One exact token-block contribution before global normalization."""

    loss_sum: torch.Tensor
    positions: int
    start: int
    stop: int


def _packed_hidden(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2]:
        raise ValueError("hidden must be [batch, positions, width] and mask [batch, positions]")
    return hidden[mask]


def iter_endpoint_kl_blocks(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    student_head: nn.Module,
    teacher_head: nn.Module,
    mask: torch.Tensor,
    *,
    token_block_size: int = 128,
    temperature: float = 1.0,
) -> Iterator[KLBlock]:
    """Yield exact endpoint/progressive forward-KL sums per token block."""

    if student_hidden.shape[:2] != teacher_hidden.shape[:2] or student_hidden.shape[:2] != mask.shape:
        raise ValueError("student, teacher, and mask position shapes must match")
    student = _packed_hidden(student_hidden, mask)
    teacher = _packed_hidden(teacher_hidden, mask).detach()
    n = student.shape[0]
    student_weight, student_bias = _linear_parts_fp32(student_head)
    with torch.no_grad():
        teacher_weight, teacher_bias = _linear_parts_fp32(teacher_head)
    for start in range(0, n, token_block_size):
        stop = min(start + token_block_size, n)
        with torch.no_grad():
            target_logits = _linear_fp32(
                teacher[start:stop], weight=teacher_weight, bias=teacher_bias
            ).detach()
        student_logits = _linear_fp32(
            student[start:stop], weight=student_weight, bias=student_bias
        )
        block_mask = torch.ones((1, stop - start), dtype=torch.bool, device=mask.device)
        loss_sum = dense_forward_kl(
            student_logits.unsqueeze(0),
            target_logits.unsqueeze(0),
            block_mask,
            temperature=temperature,
            reduction="sum",
        )
        yield KLBlock(loss_sum=loss_sum, positions=stop - start, start=start, stop=stop)


def iter_relative_kl_blocks(
    student_hidden: torch.Tensor,
    phase_reference_hidden: torch.Tensor,
    teacher_next_hidden: torch.Tensor,
    teacher_previous_hidden: torch.Tensor,
    student_head: nn.Module,
    phase_reference_head: nn.Module,
    teacher_next_head: nn.Module,
    teacher_previous_head: nn.Module,
    mask: torch.Tensor,
    *,
    token_block_size: int = 128,
    gamma: float = 1.0,
    temperature: float = 1.0,
) -> Iterator[KLBlock]:
    """Yield exact CAFD ordered-local-shift KL sums per token block."""

    position_shape = student_hidden.shape[:2]
    if any(x.shape[:2] != position_shape for x in (phase_reference_hidden, teacher_next_hidden, teacher_previous_hidden)):
        raise ValueError("all hidden-state position shapes must match")
    if mask.shape != position_shape:
        raise ValueError("mask position shape must match hidden states")

    student = _packed_hidden(student_hidden, mask)
    phase_ref = _packed_hidden(phase_reference_hidden, mask).detach()
    teacher_next = _packed_hidden(teacher_next_hidden, mask).detach()
    teacher_previous = _packed_hidden(teacher_previous_hidden, mask).detach()
    n = student.shape[0]
    student_weight, student_bias = _linear_parts_fp32(student_head)
    with torch.no_grad():
        phase_weight, phase_bias = _linear_parts_fp32(phase_reference_head)
        next_weight, next_bias = _linear_parts_fp32(teacher_next_head)
        previous_weight, previous_bias = _linear_parts_fp32(teacher_previous_head)
    for start in range(0, n, token_block_size):
        stop = min(start + token_block_size, n)
        with torch.no_grad():
            a = _linear_fp32(phase_ref[start:stop], weight=phase_weight, bias=phase_bias)
            b = _linear_fp32(teacher_next[start:stop], weight=next_weight, bias=next_bias)
            c = _linear_fp32(
                teacher_previous[start:stop], weight=previous_weight, bias=previous_bias
            )
            target_logits = relative_target_logits(a, b, c, gamma=gamma, temperature=1.0)
        student_logits = _linear_fp32(
            student[start:stop], weight=student_weight, bias=student_bias
        )
        block_mask = torch.ones((1, stop - start), dtype=torch.bool, device=mask.device)
        loss_sum = dense_forward_kl(
            student_logits.unsqueeze(0),
            target_logits.unsqueeze(0),
            block_mask,
            temperature=temperature,
            reduction="sum",
        )
        yield KLBlock(loss_sum=loss_sum, positions=stop - start, start=start, stop=stop)


def projected_endpoint_forward_kl(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    student_head: nn.Module,
    teacher_head: nn.Module,
    mask: torch.Tensor,
    *,
    token_block_size: int = 128,
    temperature: float = 1.0,
) -> torch.Tensor:
    blocks = list(
        iter_endpoint_kl_blocks(
            student_hidden,
            teacher_hidden,
            student_head,
            teacher_head,
            mask,
            token_block_size=token_block_size,
            temperature=temperature,
        )
    )
    n = sum(block.positions for block in blocks)
    if n == 0:
        return student_hidden.sum() * 0.0
    return torch.stack([block.loss_sum for block in blocks]).sum() / n


def projected_relative_forward_kl(
    student_hidden: torch.Tensor,
    phase_reference_hidden: torch.Tensor,
    teacher_next_hidden: torch.Tensor,
    teacher_previous_hidden: torch.Tensor,
    student_head: nn.Module,
    phase_reference_head: nn.Module,
    teacher_next_head: nn.Module,
    teacher_previous_head: nn.Module,
    mask: torch.Tensor,
    *,
    token_block_size: int = 128,
    gamma: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    blocks = list(
        iter_relative_kl_blocks(
            student_hidden,
            phase_reference_hidden,
            teacher_next_hidden,
            teacher_previous_hidden,
            student_head,
            phase_reference_head,
            teacher_next_head,
            teacher_previous_head,
            mask,
            token_block_size=token_block_size,
            gamma=gamma,
            temperature=temperature,
        )
    )
    n = sum(block.positions for block in blocks)
    if n == 0:
        return student_hidden.sum() * 0.0
    return torch.stack([block.loss_sum for block in blocks]).sum() / n


def assert_finite_or_dump(
    tensors: dict[str, torch.Tensor],
    *,
    batch_payload: object,
    dump_path: str,
) -> None:
    """Persist the first non-finite batch and fail; never clamp silently."""

    bad = [name for name, value in tensors.items() if not torch.isfinite(value.detach()).all()]
    if not bad:
        return
    from pathlib import Path

    path = Path(dump_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        torch.save({"non_finite": bad, "batch": batch_payload}, path)
    raise FloatingPointError(f"non-finite tensors detected: {bad}; first batch saved to {path}")
