"""Residual-anchored CAFD: detached FP32 probability mixture, exact full vocab."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .exact_forward_kl import (
    KLBlock, _packed_hidden, _linear_parts_fp32, _linear_fp32, dense_forward_kl,
)


@torch.no_grad()
def anchored_log_target(reference, next_teacher, previous_teacher):
    if reference.shape != next_teacher.shape or reference.shape != previous_teacher.shape:
        raise ValueError("all target logits must have identical shape")
    a, b, c = reference.detach().float(), next_teacher.detach().float(), previous_teacher.detach().float()
    delta, residual = b - c, a - c
    delta = delta - delta.mean(-1, keepdim=True)
    residual = residual - residual.mean(-1, keepdim=True)
    d = delta.square().mean(-1, keepdim=True).sqrt()
    r = residual.square().mean(-1, keepdim=True).sqrt()
    # Zero delta and zero residual: use the absolute Teacher; both targets coincide.
    denom = d + r
    alpha = torch.where(denom > 0, d / denom.clamp_min(torch.finfo(torch.float32).tiny), 0.0)
    log_delta = F.log_softmax(a + b - c, dim=-1)
    log_teacher = F.log_softmax(b, dim=-1)
    log_q = torch.logaddexp(log_teacher + torch.log1p(-alpha), log_delta + torch.log(alpha))
    return log_q.detach(), alpha.detach(), d.detach(), r.detach()


class AlphaStats:
    def __init__(self):
        self.total = None
        self.count = 0

    @torch.no_grad()
    def add(self, alpha, delta, residual):
        value = torch.stack((alpha.sum(), alpha.square().sum(), delta.sum(), residual.sum()))
        self.total = value if self.total is None else self.total + value
        self.count += alpha.numel()

    def result(self):
        if self.total is None or self.count == 0:
            raise RuntimeError("no completion positions measured")
        a, a2, d, r = (self.total / self.count).cpu().tolist()
        return dict(positions=self.count, alpha_mean=a, alpha_std=max(0, a2-a*a)**0.5,
                    teacher_delta_rms_mean=d, residual_rms_mean=r)


ACTIVE_STATS = None


def iter_anchored_kl_blocks(student_hidden, phase_reference_hidden,
                           teacher_next_hidden, teacher_previous_hidden,
                           student_head, phase_reference_head,
                           teacher_next_head, teacher_previous_head, mask,
                           *, token_block_size=64, gamma=1.0, temperature=1.0):
    if gamma != 1.0 or temperature != 1.0:
        raise ValueError("v14 freezes gamma=temperature=1")
    if token_block_size < 1:
        raise ValueError("positive token block size required")
    for hidden in (student_hidden, phase_reference_hidden, teacher_next_hidden, teacher_previous_hidden):
        if hidden.shape[:2] != mask.shape:
            raise ValueError("hidden/mask positions differ")
    student = _packed_hidden(student_hidden, mask)
    targets = [_packed_hidden(x, mask).detach() for x in
               (phase_reference_hidden, teacher_next_hidden, teacher_previous_hidden)]
    sw, sb = _linear_parts_fp32(student_head)
    with torch.no_grad():
        parts = [_linear_parts_fp32(h) for h in
                 (phase_reference_head, teacher_next_head, teacher_previous_head)]
    for start in range(0, student.shape[0], token_block_size):
        stop = min(start + token_block_size, student.shape[0])
        with torch.no_grad():
            logits = [_linear_fp32(h[start:stop], weight=w, bias=b)
                      for h, (w, b) in zip(targets, parts)]
            log_q, alpha, d, r = anchored_log_target(*logits)
            if ACTIVE_STATS is not None:
                ACTIVE_STATS.add(alpha, d, r)
        student_logits = _linear_fp32(student[start:stop], weight=sw, bias=sb)
        valid = torch.ones((1, stop-start), dtype=torch.bool, device=mask.device)
        loss = dense_forward_kl(student_logits.unsqueeze(0), log_q.unsqueeze(0),
                                valid, temperature=1.0, reduction="sum")
        yield KLBlock(loss_sum=loss, positions=stop-start, start=start, stop=stop)
