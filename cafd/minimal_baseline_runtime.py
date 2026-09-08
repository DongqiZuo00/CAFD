"""Absolute-target scoring, keeping validated sampling and diagnostics unchanged."""
import torch

from .kd_retention_runtime import (
    load_hidden, independent_copy, refresh_behavior, TeacherPool,
    generate, sequence_inputs, frozen_sources, target_chunks,
    restore_optimizer_exact,
)


@torch.no_grad()
def absolute_frozen_sources(record, tokenizer, teachers, coefficients, device):
    """Score only interpolated Teacher logits; neither S0 nor a subtracted T0."""
    if not coefficients or any(value < 0. for value in coefficients.values()):
        raise ValueError("absolute target requires nonnegative Teacher coefficients")
    if abs(sum(coefficients.values()) - 1.) > 1e-12:
        raise ValueError("absolute target coefficients must sum to one")
    ids, attention, mask = sequence_inputs(record, tokenizer, device)
    sources = []
    for path, coefficient in coefficients.items():
        model = teachers[path]
        hidden = model(ids, attention)[:, :-1][mask]
        head = model.lm_head
        sources.append((coefficient, hidden.detach(), head.weight.detach(),
                        None if head.bias is None else head.bias.detach()))
    return sources
