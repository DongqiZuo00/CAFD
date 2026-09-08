from __future__ import annotations

from unittest.mock import patch

import torch

from cafd.training_common import ClosingFenceEOSProcessor


class FakeStopStringCriteria:
    def __init__(self, tokenizer, strings) -> None:
        assert strings == ["\n```"]

    def __call__(self, input_ids, scores):
        del input_ids, scores
        return torch.tensor([False, True])


def test_closing_fence_forces_only_matching_rows_to_eos() -> None:
    with patch("cafd.training_common.StopStringCriteria", FakeStopStringCriteria):
        processor = ClosingFenceEOSProcessor(object(), prompt_length=2, eos_token_id=3)
    scores = torch.zeros((2, 9), dtype=torch.float32)
    result = processor(torch.tensor([[1, 1, 4], [1, 1, 5]]), scores)

    assert torch.equal(result[0], torch.zeros(9))
    assert result[1, 3].item() == 0.0
    assert torch.isneginf(result[1, torch.arange(9) != 3]).all()
