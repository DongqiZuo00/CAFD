from __future__ import annotations

import random

import numpy as np
import torch

from gap_validation.checkpoint import (
    atomic_torch_save,
    capture_rng_state,
    prune_resume_checkpoints,
    restore_rng_state,
)


def test_rng_resume_reproduces_next_draws():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), torch.rand(4))
    restore_rng_state(state)
    actual = (random.random(), float(np.random.rand()), torch.rand(4))
    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])


def test_atomic_checkpoint_and_retention(tmp_path):
    for step in (10, 20, 30):
        atomic_torch_save({"step": step}, tmp_path / f"resume-step-{step}")
    prune_resume_checkpoints(tmp_path, keep=2)
    assert not (tmp_path / "resume-step-10").exists()
    assert torch.load(tmp_path / "resume-step-20" / "state.pt", weights_only=False)["step"] == 20
    assert torch.load(tmp_path / "resume-step-30" / "state.pt", weights_only=False)["step"] == 30
