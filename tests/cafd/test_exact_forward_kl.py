from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import torch
from torch import nn

from cafd.exact_forward_kl import (
    assert_finite_or_dump,
    block_forward_kl,
    completion_prediction_mask,
    dense_forward_kl,
    iter_relative_kl_blocks,
    projected_relative_forward_kl,
)
import cafd.exact_forward_kl as exact_forward_kl
from cafd.relative_target import relative_target_logits


def test_dense_and_token_block_loss_and_student_logits_gradient_match() -> None:
    torch.manual_seed(11)
    student_dense = torch.randn(3, 9, 31, requires_grad=True)
    student_block = student_dense.detach().clone().requires_grad_(True)
    target = torch.randn(3, 9, 31)
    mask = torch.rand(3, 9) > 0.35
    dense = dense_forward_kl(student_dense, target, mask)
    block = block_forward_kl(student_block, target, mask, token_block_size=4)
    dense.backward()
    block.backward()
    torch.testing.assert_close(block, dense, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(student_block.grad, student_dense.grad, atol=5e-5, rtol=5e-5)
    cosine = torch.nn.functional.cosine_similarity(student_block.grad.flatten(), student_dense.grad.flatten(), dim=0)
    assert cosine >= 0.999999


def test_exact_student_logit_gradient_is_p_minus_q_over_n() -> None:
    torch.manual_seed(12)
    student = torch.randn(2, 4, 17, requires_grad=True)
    target = torch.randn(2, 4, 17)
    mask = torch.tensor([[True, False, True, True], [False, True, True, False]])
    loss = block_forward_kl(student, target, mask, token_block_size=2)
    loss.backward()
    n = int(mask.sum())
    expected = torch.zeros_like(student)
    expected[mask] = (student.detach()[mask].softmax(-1) - target[mask].softmax(-1)) / n
    torch.testing.assert_close(student.grad, expected, atol=5e-5, rtol=5e-5)


class TinyBackbone(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(torch.tanh(self.in_proj(x)))


def _run_projected(block_size: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
    torch.manual_seed(13)
    batch, positions, width, vocab = 2, 6, 8, 29
    backbone = TinyBackbone(width)
    head = nn.Linear(width, vocab, bias=False)
    x = torch.randn(batch, positions, width)
    student_hidden = backbone(x)
    phase = torch.randn_like(student_hidden)
    next_teacher = torch.randn_like(student_hidden)
    previous = torch.randn_like(student_hidden)
    phase_head = nn.Linear(width, vocab, bias=False)
    next_head = nn.Linear(width, vocab, bias=False)
    previous_head = nn.Linear(width, vocab, bias=False)
    mask = torch.tensor([[True, True, False, True, False, True], [False, True, True, True, True, False]])
    loss = projected_relative_forward_kl(
        student_hidden,
        phase,
        next_teacher,
        previous,
        head,
        phase_head,
        next_head,
        previous_head,
        mask,
        token_block_size=block_size,
    )
    loss.backward()
    grads = [parameter.grad.detach().clone() for parameter in [*backbone.parameters(), *head.parameters()]]
    return loss.detach(), grads


def test_lm_head_hidden_and_upstream_gradients_match_dense_projection() -> None:
    dense_loss, dense_grads = _run_projected(10_000)
    block_loss, block_grads = _run_projected(3)
    torch.testing.assert_close(block_loss, dense_loss, atol=2e-5, rtol=2e-5)
    for actual, expected in zip(block_grads, dense_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)
        cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
        assert cosine >= 0.999999


def test_full_lm_heads_are_materialized_once_per_block_iterator() -> None:
    torch.manual_seed(131)
    batch, positions, width, vocab = 2, 7, 5, 23
    hidden = torch.randn(batch, positions, width, requires_grad=True)
    targets = [torch.randn(batch, positions, width) for _ in range(3)]
    heads = [nn.Linear(width, vocab, bias=False) for _ in range(4)]
    mask = torch.ones(batch, positions, dtype=torch.bool)

    with patch(
        "cafd.exact_forward_kl._linear_parts_fp32",
        wraps=exact_forward_kl._linear_parts_fp32,
    ) as materialize:
        blocks = list(
            iter_relative_kl_blocks(
                hidden,
                targets[0],
                targets[1],
                targets[2],
                heads[0],
                heads[1],
                heads[2],
                heads[3],
                mask,
                token_block_size=3,
            )
        )

    assert len(blocks) > 1
    assert materialize.call_count == 4
    torch.stack([block.loss_sum for block in blocks]).sum().backward()
    assert heads[0].weight.grad is not None
    assert all(head.weight.grad is None for head in heads[1:])


def test_bfloat16_projection_with_fp32_accumulation() -> None:
    if not hasattr(torch, "bfloat16"):
        pytest.skip("bfloat16 unavailable")
    torch.manual_seed(14)
    student = torch.randn(2, 5, 37, dtype=torch.float32, requires_grad=True)
    target = torch.randn(2, 5, 37, dtype=torch.float32)
    mask = torch.rand(2, 5) > 0.2
    fp32 = dense_forward_kl(student, target, mask)
    fp32.backward()
    reference_grad = student.grad.detach().clone()
    bf16 = student.detach().to(torch.bfloat16).requires_grad_(True)
    approx = block_forward_kl(bf16, target.to(torch.bfloat16), mask, token_block_size=3)
    approx.backward()
    relative_error = abs(float(approx - fp32.detach())) / max(abs(float(fp32)), 1e-8)
    cosine = torch.nn.functional.cosine_similarity(bf16.grad.float().flatten(), reference_grad.flatten(), dim=0)
    assert relative_error <= 5e-3
    assert cosine >= 0.999


def test_completion_mask_prompt_padding_eos_and_empty_completion() -> None:
    eos, pad = 2, 0
    input_ids = torch.tensor(
        [
            [8, 9, 10, 20, 21, eos, pad, pad],
            [7, 30, eos, 44, pad, pad, pad, pad],
            [5, 6, 7, pad, pad, pad, pad, pad],
            [4, 41, 42, 43, 44, pad, pad, pad],
        ]
    )
    attention = input_ids.ne(pad)
    prompt_lengths = torch.tensor([3, 1, 3, 2])
    mask = completion_prediction_mask(input_ids, attention, prompt_lengths, eos_token_id=eos)
    expected = torch.tensor(
        [
            [False, False, True, True, True, False, False],
            [True, True, False, False, False, False, False],
            [False, False, False, False, False, False, False],
            [False, True, True, True, False, False, False],
        ]
    )
    assert torch.equal(mask.cpu(), expected)


def test_nonfinite_batch_is_saved_once_and_raises(tmp_path) -> None:
    path = tmp_path / "first_failure.pt"
    with pytest.raises(FloatingPointError):
        assert_finite_or_dump({"loss": torch.tensor(float("nan"))}, batch_payload={"id": 7}, dump_path=str(path))
    assert path.exists()
    original = path.read_bytes()
    with pytest.raises(FloatingPointError):
        assert_finite_or_dump({"loss": torch.tensor(float("inf"))}, batch_payload={"id": 8}, dump_path=str(path))
    assert path.read_bytes() == original
