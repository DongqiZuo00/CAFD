from __future__ import annotations

import json

import torch

from gap_validation.kl import dense_forward_kl, exact_chunked_forward_kl
from trl.trainer.distillation_trainer import _chunked_divergence_loss


def test_exact_chunked_forward_kl_matches_dense(tmp_path):
    generator = torch.Generator().manual_seed(42)
    teacher = torch.randn(2, 5, 37, generator=generator, dtype=torch.float64)
    student_dense = torch.randn(2, 5, 37, generator=generator, dtype=torch.float64, requires_grad=True)
    student_chunked = student_dense.detach().clone().requires_grad_(True)
    mask = torch.tensor([[0, 1, 1, 1, 0], [0, 0, 1, 1, 1]], dtype=torch.bool)

    dense = dense_forward_kl(student_dense, teacher, mask, temperature=1.3)
    chunked = exact_chunked_forward_kl(
        student_chunked, teacher, mask, temperature=1.3, vocab_chunk_size=7
    )
    dense.backward()
    chunked.backward()

    dense_grad = student_dense.grad.detach()
    chunked_grad = student_chunked.grad.detach()
    absolute_loss_error = float((dense - chunked).abs())
    relative_loss_error = absolute_loss_error / max(float(dense.abs()), 1e-30)
    cosine = float(torch.nn.functional.cosine_similarity(dense_grad.flatten(), chunked_grad.flatten(), dim=0))
    maximum_gradient_error = float((dense_grad - chunked_grad).abs().max())
    result = {
        "absolute_loss_error": absolute_loss_error,
        "relative_loss_error": relative_loss_error,
        "gradient_cosine_similarity": cosine,
        "maximum_gradient_error": maximum_gradient_error,
    }
    (tmp_path / "kl_alignment.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    assert absolute_loss_error < 2e-7
    assert relative_loss_error < 2e-6
    assert cosine > 0.999999
    assert maximum_gradient_error < 2e-7


def test_chunked_kl_rejects_invalid_inputs():
    left = torch.zeros(1, 1, 3)
    right = torch.zeros(1, 1, 4)
    try:
        exact_chunked_forward_kl(left, right)
    except ValueError as error:
        assert "identical shapes" in str(error)
    else:
        raise AssertionError("shape mismatch was accepted")


def test_published_trl_forward_kl_matches_dense_reference():
    generator = torch.Generator().manual_seed(42)
    student_hidden = torch.randn(2, 4, 7, generator=generator, requires_grad=True)
    teacher_hidden = torch.randn(2, 4, 11, generator=generator)
    student_head = torch.randn(31, 7, generator=generator, requires_grad=True)
    teacher_head = torch.randn(31, 11, generator=generator)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

    reference_logits = student_hidden @ student_head.t()
    teacher_logits = teacher_hidden @ teacher_head.t()
    dense = dense_forward_kl(reference_logits, teacher_logits, mask, temperature=1.0)
    dense_grad_hidden, dense_grad_head = torch.autograd.grad(dense, (student_hidden, student_head))

    student_hidden_trl = student_hidden.detach().clone().requires_grad_(True)
    student_head_trl = student_head.detach().clone().requires_grad_(True)
    trl_loss, _, _ = _chunked_divergence_loss(
        student_hidden_trl,
        teacher_hidden,
        student_head_trl,
        teacher_head,
        mask,
        beta=0.0,
        chunk_size=3,
        temperature=1.0,
    )
    trl_grad_hidden, trl_grad_head = torch.autograd.grad(trl_loss, (student_hidden_trl, student_head_trl))

    assert torch.allclose(trl_loss, dense, atol=2e-6, rtol=2e-6)
    assert torch.allclose(trl_grad_hidden, dense_grad_hidden, atol=2e-6, rtol=2e-5)
    assert torch.allclose(trl_grad_head, dense_grad_head, atol=2e-6, rtol=2e-5)
