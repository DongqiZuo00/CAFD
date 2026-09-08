from __future__ import annotations

import torch


def _masked_token_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask = mask.to(dtype=values.dtype)
    denominator = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denominator


def dense_forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    completion_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("teacher and student logits must have identical shapes")
    teacher = teacher_logits.detach().float() / temperature
    student = student_logits.float() / temperature
    teacher_logp = torch.log_softmax(teacher, dim=-1)
    student_logp = torch.log_softmax(student, dim=-1)
    per_token = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1)
    return _masked_token_mean(per_token, completion_mask) * (temperature**2)


def _chunked_logsumexp(logits: torch.Tensor, chunk_size: int) -> torch.Tensor:
    total: torch.Tensor | None = None
    for start in range(0, logits.shape[-1], chunk_size):
        current = torch.logsumexp(logits[..., start : start + chunk_size], dim=-1)
        total = current if total is None else torch.logaddexp(total, current)
    if total is None:
        raise ValueError("empty vocabulary")
    return total


def exact_chunked_forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    completion_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    vocab_chunk_size: int = 8192,
) -> torch.Tensor:
    """Exact full-vocabulary KL(stopgrad(teacher) || student).

    Each distribution uses one global log-normalizer accumulated across all
    vocabulary chunks. No chunk-local softmax, top-k, or sampled vocabulary is used.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("teacher and student logits must have identical shapes")
    if vocab_chunk_size <= 0:
        raise ValueError("vocab_chunk_size must be positive")
    teacher = teacher_logits.detach().float() / temperature
    student = student_logits.float() / temperature
    teacher_lse = _chunked_logsumexp(teacher, vocab_chunk_size)
    student_lse = _chunked_logsumexp(student, vocab_chunk_size)
    per_token = torch.zeros_like(teacher_lse)
    for start in range(0, teacher.shape[-1], vocab_chunk_size):
        teacher_chunk = teacher[..., start : start + vocab_chunk_size]
        student_chunk = student[..., start : start + vocab_chunk_size]
        teacher_logp = teacher_chunk - teacher_lse.unsqueeze(-1)
        student_logp = student_chunk - student_lse.unsqueeze(-1)
        per_token = per_token + (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1)
    return _masked_token_mean(per_token, completion_mask) * (temperature**2)

