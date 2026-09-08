from __future__ import annotations

from types import SimpleNamespace
import copy

import torch
from torch import nn

from cafd.exact_forward_kl import iter_endpoint_kl_blocks
from cafd.optimizer import FP32AdamW
from cafd.train_student import _distillation_update, _make_sequence_batch
from cafd.training_common import make_sequence


class TinyHiddenLM(nn.Module):
    def __init__(self, vocab: int = 43, width: int = 7) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.backbone = nn.Linear(width, width)
        self.lm_head = nn.Linear(width, vocab, bias=False)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        return torch.tanh(self.backbone(self.embedding(input_ids)))


def test_distillation_microbatch_preserves_each_completion_mask() -> None:
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    rollouts = [
        {"prompt_ids": [8, 9, 10], "completion_ids": [20, 21, 2]},
        {"prompt_ids": [7], "completion_ids": [30, 2]},
        {"prompt_ids": [5, 6], "completion_ids": [40, 41, 42, 2]},
    ]

    input_ids, attention, mask = _make_sequence_batch(rollouts, tokenizer, torch.device("cpu"))

    assert input_ids.shape == (3, 6)
    assert attention.sum(dim=1).tolist() == [6, 3, 6]
    assert mask.sum(dim=1).tolist() == [3, 2, 4]
    assert input_ids[1].tolist() == [7, 30, 2, 0, 0, 0]


def test_batched_endpoint_update_matches_sequence_reference(tmp_path) -> None:
    torch.manual_seed(212)
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    rollouts = [
        {"row_id": str(index), "prompt_ids": [8 + index, 12], "completion_ids": completion}
        for index, completion in enumerate(([20, 21, 2], [22, 2], [23, 24, 25, 2], [26, 27, 2]))
    ]
    initial = TinyHiddenLM()
    batched = copy.deepcopy(initial)
    reference = copy.deepcopy(initial)
    teacher = TinyHiddenLM().eval().requires_grad_(False)
    teacher_batched = copy.deepcopy(teacher)
    batched_optimizer = FP32AdamW(batched.parameters(), lr=1e-4)
    reference_optimizer = FP32AdamW(reference.parameters(), lr=1e-4)

    positions = sum(len(item["completion_ids"]) for item in rollouts)
    reference_optimizer.zero_grad(set_to_none=True)
    for item in rollouts:
        ids, attention, _, mask = make_sequence(
            item["prompt_ids"], item["completion_ids"], tokenizer, torch.device("cpu")
        )
        student_hidden = reference(ids, attention)[:, :-1]
        with torch.no_grad():
            teacher_hidden = teacher(ids, attention)[:, :-1]
        loss = torch.stack(
            [
                block.loss_sum
                for block in iter_endpoint_kl_blocks(
                    student_hidden,
                    teacher_hidden,
                    reference.lm_head,
                    teacher.lm_head,
                    mask,
                    token_block_size=2,
                )
            ]
        ).sum()
        (loss / positions).backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    reference_optimizer.step()

    loss, scored, gradient_norm = _distillation_update(
        "endpoint",
        batched,
        batched_optimizer,
        tokenizer,
        rollouts,
        {"R5": teacher_batched},
        None,
        0,
        2,
        tmp_path,
    )

    assert loss > 0.0
    assert scored == positions
    assert gradient_norm > 0.0

    for actual, expected in zip(batched.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
