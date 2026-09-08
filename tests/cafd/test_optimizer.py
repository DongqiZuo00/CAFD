from __future__ import annotations

import torch

from cafd.optimizer import FP32AdamW


def test_fp32_master_accumulates_updates_below_bf16_resolution() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    optimizer = FP32AdamW([parameter], lr=1.0e-4, betas=(0.0, 0.0), eps=0.0)

    for _ in range(100):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()

    state = optimizer.state[parameter]
    assert state["master"].dtype == torch.float32
    torch.testing.assert_close(state["master"], torch.tensor([0.99]), atol=2e-6, rtol=0.0)
    assert parameter.item() < 0.995
    optimizer.assert_fp32_states()


def test_legacy_state_initializes_master_once() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    optimizer = FP32AdamW([parameter], lr=1.0e-4, betas=(0.0, 0.0), eps=0.0)
    optimizer.state[parameter] = {
        "step": 7,
        "exp_avg": torch.zeros(1, dtype=torch.float32),
        "exp_avg_sq": torch.zeros(1, dtype=torch.float32),
    }

    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    assert optimizer.state[parameter]["master"].dtype == torch.float32
    assert optimizer.state[parameter]["step"] == 8
    optimizer.assert_fp32_states()
