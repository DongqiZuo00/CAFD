"""CAFD-MPC exact mathematical objective; independent of legacy CAFD targets.

All softmaxes cover the full vocabulary in FP32, temperature one. The streaming
linear projections recompute their distributions in backward rather than retain
a completion-length by vocabulary activation. This costs a second projection,
not an approximation. Only the Student hidden states and LM head get gradients.
"""
from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from .exact_forward_kl import completion_prediction_mask

TargetSource: TypeAlias = tuple[float, torch.Tensor, torch.Tensor, torch.Tensor | None]


def teacher_coefficients(u: float, route_ids: Sequence[Hashable]) -> dict[Hashable, float]:
    """Coalesce (1-alpha)*T_m + alpha*T_(m+1) - T_0 by checkpoint ID."""
    m_max = len(route_ids) - 1
    if m_max < 1:
        raise ValueError("Teacher route needs at least two checkpoints")
    u = float(u)
    if not math.isfinite(u) or not 0.0 <= u <= m_max:
        raise ValueError("u must be finite and in [0, M]")
    m = min(math.floor(u), m_max - 1)
    alpha = u - m
    result: dict[Hashable, float] = {}
    for checkpoint, coefficient in (
        (route_ids[0], -1.0), (route_ids[m], 1.0 - alpha),
        (route_ids[m + 1], alpha),
    ):
        result[checkpoint] = result.get(checkpoint, 0.0) + coefficient
    return {checkpoint: coefficient for checkpoint, coefficient in result.items() if coefficient != 0.0}


@torch.no_grad()
def cumulative_target_logits(
    student0_logits: torch.Tensor,
    teacher_logits_by_id: Mapping[Hashable, torch.Tensor],
    u: float,
    route_ids: Sequence[Hashable],
) -> torch.Tensor:
    """Detached dense reference. Only nonzero, deduplicated Teacher terms needed."""
    target = student0_logits.detach().float().clone()
    for checkpoint, coefficient in teacher_coefficients(u, route_ids).items():
        other = teacher_logits_by_id[checkpoint]
        if other.shape != target.shape or other.device != target.device:
            raise ValueError("All models must score identical prefixes and vocabulary")
        target.add_(other.detach().float(), alpha=coefficient)
    return target


@torch.no_grad()
def cumulative_target_probs(
    student0_logits: torch.Tensor,
    teacher_logits_by_id: Mapping[Hashable, torch.Tensor],
    u: float,
    route_ids: Sequence[Hashable],
) -> torch.Tensor:
    return F.softmax(cumulative_target_logits(student0_logits, teacher_logits_by_id, u, route_ids), dim=-1)


@dataclass(frozen=True)
class RewardRouting:
    distill: torch.Tensor
    rl: torch.Tensor
    skip: torch.Tensor
    advantages: torch.Tensor

    @property
    def should_step(self) -> bool:
        """False requires skipping the entire optimizer step, not merely backward."""
        return bool((self.distill | self.rl).any().item())


@torch.no_grad()
def route_reward_groups(
    rewards: torch.Tensor,
    full_pass: torch.Tensor | None = None,
    *,
    advantage_epsilon: float = 1e-6,
) -> RewardRouting:
    """Route prompt groups [P,R], with population-standard-deviation advantage."""
    # Do not round a provided FP64 near-one failure into reward==1, or erase
    # nonzero within-group variation before making the exact routing decision.
    rewards = rewards.detach().to(dtype=torch.float64 if rewards.dtype == torch.float64 else torch.float32)
    if rewards.ndim != 2 or rewards.shape[0] < 1 or rewards.shape[1] < 2:
        raise ValueError("rewards must have shape [P,R], P>=1 and R>=2")
    if not bool(torch.isfinite(rewards).all()) or bool(((rewards < 0) | (rewards > 1)).any()):
        raise ValueError("Verifier rewards must be finite in [0,1]")
    if not math.isfinite(advantage_epsilon) or advantage_epsilon <= 0:
        raise ValueError("advantage_epsilon must be positive and finite")
    expected_full_pass = rewards == 1.0
    if full_pass is not None:
        if full_pass.shape != rewards.shape:
            raise ValueError("full_pass and rewards shapes must match")
        full_pass = full_pass.detach().to(device=rewards.device)
        if not bool(((full_pass == 0) | (full_pass == 1)).all()):
            raise ValueError("full_pass must be a boolean/0-1 array")
        if not torch.equal(full_pass.bool(), expected_full_pass):
            raise ValueError("Verifier full_pass must be exactly equivalent to reward==1")
    spread = rewards.amax(dim=-1) - rewards.amin(dim=-1)
    rl = spread > 0.0
    distill = (spread == 0.0) & ~expected_full_pass.any(dim=-1)
    skip = ~(rl | distill)
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    sigma = centered.square().mean(dim=-1, keepdim=True).sqrt()
    advantages = centered / (sigma + advantage_epsilon)
    advantages = torch.where(rl[:, None], advantages, torch.zeros_like(advantages))
    return RewardRouting(distill, rl, skip, advantages.float())


def _normalizer(value: float | int | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.requires_grad:
            raise ValueError("normalization_tokens must be a detached scalar")
        value = value.item()
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("normalization_tokens must be positive and finite")
    return value


def _broadcast_advantages(advantages: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    advantages = advantages.detach().float()
    while advantages.ndim < len(shape):
        advantages = advantages.unsqueeze(-1)
    return torch.broadcast_to(advantages, shape)


def clipped_rl_loss(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    normalization_tokens: float | int | torch.Tensor,
    clip: float = 0.2,
    lambda_rl: float = 1.0,
) -> torch.Tensor:
    """Tokenwise clipped group-relative RL with shared whole-batch token Z."""
    if current_log_probs.shape != old_log_probs.shape or current_log_probs.shape != mask.shape:
        raise ValueError("current/old token log probabilities and mask shapes must match")
    if not math.isfinite(clip) or not 0 <= clip < 1:
        raise ValueError("clip must be finite in [0,1)")
    if not math.isfinite(lambda_rl) or lambda_rl < 0:
        raise ValueError("lambda_rl must be nonnegative and finite")
    denom = _normalizer(normalization_tokens)
    valid = mask.bool()
    new = current_log_probs[valid].float()
    old = old_log_probs.detach()[valid].float()
    adv = _broadcast_advantages(advantages, mask.shape)[valid]
    ratio = (new - old).exp()
    surrogate = torch.minimum(ratio * adv, ratio.clamp(1.0 - clip, 1.0 + clip) * adv)
    return -float(lambda_rl) * surrogate.sum() / denom


@dataclass(frozen=True)
class MixedObjective:
    loss: torch.Tensor
    distillation_loss: torch.Tensor
    rl_loss: torch.Tensor
    normalization_tokens: int
    routing: RewardRouting

    @property
    def should_step(self) -> bool:
        return self.routing.should_step and self.normalization_tokens > 0


def mixed_objective_reference(
    student_logits: torch.Tensor,
    target_logits: torch.Tensor | None,
    old_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    completion_mask: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    *,
    full_pass: torch.Tensor | None = None,
    clip: float = 0.2,
    lambda_rl: float = 1.0,
    advantage_epsilon: float = 1e-6,
) -> MixedObjective:
    """Dense test/reference implementation; production should use linear streaming.

    Expected shapes are student/target [P,R,L,V], mask/ids/old_logp [P,R,L].
    target_logits may be None when no prompt group is routed to distillation.
    Its entries for non-distillation groups are never read.
    """
    if student_logits.ndim != 4 or completion_mask.shape != student_logits.shape[:-1]:
        raise ValueError("Student logits must be [P,R,L,V] with matching mask")
    if sampled_token_ids.shape != completion_mask.shape or old_log_probs.shape != completion_mask.shape:
        raise ValueError("Sampled IDs and old log probabilities must match mask")
    if rewards.shape != student_logits.shape[:2]:
        raise ValueError("Rewards shape must match prompt/rollout axes")
    routing = route_reward_groups(rewards, full_pass, advantage_epsilon=advantage_epsilon)
    mask = completion_mask.bool()
    z = int(mask.sum().item())  # Includes all-success zero-gradient groups.
    zero = student_logits[..., :0].sum().float()
    if z == 0:
        return MixedObjective(zero, zero, zero, 0, routing)
    d_mask = mask & routing.distill[:, None, None]
    r_mask = mask & routing.rl[:, None, None]
    d_loss = zero
    if bool(d_mask.any()):
        if target_logits is None or target_logits.shape != student_logits.shape:
            raise ValueError("Distillation needs full-vocabulary target logits")
        log_q = F.log_softmax(target_logits.detach()[d_mask].float(), dim=-1)
        log_p = F.log_softmax(student_logits[d_mask].float(), dim=-1)
        d_loss = (log_q.exp() * (log_q - log_p)).sum() / z
    r_loss = zero
    if bool(r_mask.any()):
        selected = F.log_softmax(student_logits[r_mask].float(), dim=-1)
        ids = sampled_token_ids[r_mask].long()
        new_logp = selected.gather(-1, ids[:, None]).squeeze(-1)
        adv = _broadcast_advantages(routing.advantages, mask.shape)[r_mask]
        r_loss = clipped_rl_loss(
            new_logp, old_log_probs[r_mask], adv,
            torch.ones_like(new_logp, dtype=torch.bool),
            normalization_tokens=z, clip=clip, lambda_rl=lambda_rl,
        )
    return MixedObjective(d_loss + r_loss, d_loss, r_loss, z, routing)


def _validate_projection(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    token_chunk: int,
) -> None:
    if hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1]:
        raise ValueError("Projection requires hidden[N,H], weight[V,H]")
    if not hidden.is_floating_point() or not weight.is_floating_point():
        raise ValueError("Projection tensors must be floating point")
    if hidden.device != weight.device:
        raise ValueError("Hidden and projection weight must be on the same device")
    if bias is not None and (bias.shape != weight.shape[:1] or bias.device != hidden.device):
        raise ValueError("Projection bias must be [V] on the same device")
    if isinstance(token_chunk, bool) or not isinstance(token_chunk, int) or token_chunk < 1:
        raise ValueError("token_chunk must be a positive integer")


def _project(hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    # Explicitly disable caller autocast: method contract is FP32 full-vocabulary.
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        return F.linear(hidden.float(), weight.float(), None if bias is None else bias.float())


def _target_block(sources: Sequence[TargetSource], start: int, stop: int) -> torch.Tensor:
    target = None
    for coefficient, hidden, weight, bias in sources:
        term = _project(hidden[start:stop], weight, bias)
        if target is None:
            target = term.mul(coefficient)
        else:
            target.add_(term, alpha=coefficient)
    assert target is not None
    return target


class _LinearForwardKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, sources, normalization_tokens, token_chunk):
        saved = [hidden, weight]
        ctx.has_bias = bias is not None
        if bias is not None:
            saved.append(bias)
        ctx.source_layout = []
        for coefficient, sh, sw, sb in sources:
            ctx.source_layout.append((coefficient, sb is not None))
            saved.extend([sh, sw])
            if sb is not None:
                saved.append(sb)
        ctx.save_for_backward(*saved)
        ctx.normalizer = normalization_tokens
        ctx.token_chunk = token_chunk
        total = torch.zeros((), dtype=torch.float32, device=hidden.device)
        for start in range(0, hidden.shape[0], token_chunk):
            stop = min(start + token_chunk, hidden.shape[0])
            log_p = F.log_softmax(_project(hidden[start:stop], weight, bias), dim=-1)
            log_q = F.log_softmax(_target_block(sources, start, stop), dim=-1)
            total.add_((log_q.exp() * (log_q - log_p)).sum())
        return total / normalization_tokens

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        hidden, weight = saved[:2]
        cursor = 2
        bias = saved[cursor] if ctx.has_bias else None
        cursor += int(ctx.has_bias)
        sources = []
        for coefficient, has_bias in ctx.source_layout:
            sh, sw = saved[cursor:cursor + 2]
            cursor += 2
            sb = saved[cursor] if has_bias else None
            cursor += int(has_bias)
            sources.append((coefficient, sh, sw, sb))
        gh = torch.zeros_like(hidden, dtype=torch.float32) if ctx.needs_input_grad[0] else None
        gw = torch.zeros_like(weight, dtype=torch.float32) if ctx.needs_input_grad[1] else None
        gb = torch.zeros_like(bias, dtype=torch.float32) if ctx.has_bias and ctx.needs_input_grad[2] else None
        scale = grad_output.float() / ctx.normalizer
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            for start in range(0, hidden.shape[0], ctx.token_chunk):
                stop = min(start + ctx.token_chunk, hidden.shape[0])
                p = F.softmax(_project(hidden[start:stop], weight, bias), dim=-1)
                q = F.softmax(_target_block(sources, start, stop), dim=-1)
                dz = (p - q).mul_(scale)
                if gh is not None:
                    gh[start:stop] = dz @ weight.float()
                if gw is not None:
                    gw.addmm_(dz.transpose(0, 1), hidden[start:stop].float())
                if gb is not None:
                    gb.add_(dz.sum(dim=0))
        return (
            None if gh is None else gh.to(hidden.dtype),
            None if gw is None else gw.to(weight.dtype),
            None if gb is None else gb.to(bias.dtype),
            None, None, None,
        )


def exact_linear_forward_kl(
    student_hidden: torch.Tensor,
    student_weight: torch.Tensor,
    target_sources: Sequence[TargetSource],
    *,
    normalization_tokens: float | int | torch.Tensor,
    token_chunk: int = 64,
    student_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact KL sum / whole-batch Z, streaming full-vocabulary token chunks.

    Caller supplies ONLY valid completion positions of distillation-routed groups.
    Sources are (coefficient, frozen_hidden[N,H_i], frozen_head[V,H_i], bias|None):
    one permanent-S0 source with coefficient +1 plus nonzero deduplicated Teacher
    sources. Different Student/Teacher hidden widths are supported; vocab must
    match exactly. Tensors are detached internally even if caller forgot.

    Peak distribution activation is O(token_chunk * V), not O(N * V). This also
    retains O(N * sum(H_i)) hidden states and an FP32 full Student-head gradient;
    each BF16 projection head is temporarily converted to FP32. No top-k,
    vocabulary sampling, cached full q, or gradient reaches a target source.
    """
    _validate_projection(student_hidden, student_weight, student_bias, token_chunk)
    z = _normalizer(normalization_tokens)
    sources = []
    for coefficient, hidden, weight, bias in target_sources:
        coefficient = float(coefficient)
        if not math.isfinite(coefficient):
            raise ValueError("Target coefficients must be finite")
        if coefficient == 0.0:
            continue
        _validate_projection(hidden, weight, bias, token_chunk)
        if hidden.shape[0] != student_hidden.shape[0] or weight.shape[0] != student_weight.shape[0]:
            raise ValueError("Target prefixes and full vocabulary must match Student")
        if hidden.device != student_hidden.device:
            raise ValueError("All target sources must be on the Student device")
        sources.append((
            coefficient, hidden.detach(), weight.detach(),
            None if bias is None else bias.detach(),
        ))
    if not sources:
        raise ValueError("At least permanent S0 target source is required")
    return _LinearForwardKL.apply(student_hidden, student_weight, student_bias, tuple(sources), z, token_chunk)


class _LinearTokenLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, token_ids, token_chunk):
        ctx.has_bias = bias is not None
        ctx.save_for_backward(hidden, weight, token_ids, *([] if bias is None else [bias]))
        ctx.token_chunk = token_chunk
        output = torch.empty(hidden.shape[0], dtype=torch.float32, device=hidden.device)
        for start in range(0, hidden.shape[0], token_chunk):
            stop = min(start + token_chunk, hidden.shape[0])
            log_p = F.log_softmax(_project(hidden[start:stop], weight, bias), dim=-1)
            output[start:stop] = log_p.gather(-1, token_ids[start:stop, None]).squeeze(-1)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        hidden, weight, token_ids, *bias_list = ctx.saved_tensors
        bias = bias_list[0] if ctx.has_bias else None
        gh = torch.zeros_like(hidden, dtype=torch.float32) if ctx.needs_input_grad[0] else None
        gw = torch.zeros_like(weight, dtype=torch.float32) if ctx.needs_input_grad[1] else None
        gb = torch.zeros_like(bias, dtype=torch.float32) if ctx.has_bias and ctx.needs_input_grad[2] else None
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            for start in range(0, hidden.shape[0], ctx.token_chunk):
                stop = min(start + ctx.token_chunk, hidden.shape[0])
                dz = -F.softmax(_project(hidden[start:stop], weight, bias), dim=-1)
                dz.scatter_add_(1, token_ids[start:stop, None], torch.ones(stop - start, 1, device=hidden.device))
                dz.mul_(grad_output[start:stop, None].float())
                if gh is not None:
                    gh[start:stop] = dz @ weight.float()
                if gw is not None:
                    gw.addmm_(dz.transpose(0, 1), hidden[start:stop].float())
                if gb is not None:
                    gb.add_(dz.sum(dim=0))
        return (
            None if gh is None else gh.to(hidden.dtype),
            None if gw is None else gw.to(weight.dtype),
            None if gb is None else gb.to(bias.dtype),
            None, None,
        )


def exact_linear_token_log_probs(
    student_hidden: torch.Tensor,
    student_weight: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    token_chunk: int = 64,
    student_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full-vocabulary log p(sampled token), with recomputed exact backward.

    This is the unwarped policy probability. The runner must ensure the logged
    old behavior probabilities correspond to its sampling distribution; do not
    reuse top-p/temperature warped probabilities as if they were unwarped.
    """
    _validate_projection(student_hidden, student_weight, student_bias, token_chunk)
    if token_ids.shape != student_hidden.shape[:1] or token_ids.device != student_hidden.device:
        raise ValueError("token_ids must be [N] on the Student device")
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("token_ids must be integer IDs")
    token_ids = token_ids.detach().long()
    if bool(((token_ids < 0) | (token_ids >= student_weight.shape[0])).any()):
        raise ValueError("token IDs outside full vocabulary")
    return _LinearTokenLogProbs.apply(student_hidden, student_weight, student_bias, token_ids, token_chunk)
