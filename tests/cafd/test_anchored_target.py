import copy
from pathlib import Path
import pytest
import torch
from torch import nn
from cafd.anchored_target import anchored_log_target, iter_anchored_kl_blocks
from cafd.exact_forward_kl import dense_forward_kl
from cafd.data import prompt_stream
from cafd.mistral_anchored_v14 import resolved_config


def test_probability_mixture_and_offset_invariance():
    torch.manual_seed(2027)
    a, b, c = [torch.randn(7, 19, requires_grad=True) for _ in range(3)]
    logq, alpha, d, r = anchored_log_target(a, b, c)
    expected = (1-alpha)*b.softmax(-1)+alpha*(a+b-c).softmax(-1)
    torch.testing.assert_close(logq.exp(), expected)
    torch.testing.assert_close(logq.exp().sum(-1), torch.ones(7))
    shifted = anchored_log_target(a+3, b-4, c+5)
    torch.testing.assert_close(shifted[0], logq, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(shifted[1], alpha, atol=3e-6, rtol=3e-6)
    assert all(not x.requires_grad for x in (logq, alpha, d, r))


def test_zero_delta_zero_residual_and_extreme_logits():
    a, b, c = [torch.randn(4, 23) for _ in range(3)]
    logq, alpha, *_ = anchored_log_target(a, c, c)
    assert torch.equal(alpha, torch.zeros_like(alpha))
    torch.testing.assert_close(logq.exp(), c.softmax(-1))
    logq, alpha, *_ = anchored_log_target(c, b, c)
    torch.testing.assert_close(logq.exp(), b.softmax(-1))
    logq, *_ = anchored_log_target(c, c, c)
    assert torch.isfinite(logq).all()
    logq, *_ = anchored_log_target(a*10000, b*10000, c*10000)
    assert torch.isfinite(logq).all()
    torch.testing.assert_close(logq.exp().sum(-1), torch.ones(4), atol=1e-5, rtol=1e-5)


def test_full_vocabulary_block_loss_and_gradients_match_dense():
    torch.manual_seed(7)
    hidden = torch.randn(2, 5, 6, requires_grad=True)
    refs = [torch.randn(2, 5, 6, requires_grad=True) for _ in range(3)]
    heads = [nn.Linear(6, 31) for _ in range(4)]
    mask = torch.tensor([[1,1,0,1,0],[1,0,1,1,1]], dtype=torch.bool)
    loss = sum(x.loss_sum for x in iter_anchored_kl_blocks(
        hidden, *refs, *heads, mask, token_block_size=2)) / mask.sum()
    loss.backward()
    hidden2 = hidden.detach().clone().requires_grad_()
    head2 = copy.deepcopy(heads[0])
    head2.zero_grad()
    logq, *_ = anchored_log_target(*[h(x) for h,x in zip(heads[1:],refs)])
    dense = dense_forward_kl(head2(hidden2), logq, mask)
    dense.backward()
    torch.testing.assert_close(loss, dense)
    torch.testing.assert_close(hidden.grad, hidden2.grad)
    torch.testing.assert_close(heads[0].weight.grad, head2.weight.grad)
    assert all(x.grad is None for x in refs)
    assert all(p.grad is None for h in heads[1:] for p in h.parameters())


def test_pilot_keeps_prompt_stream_prefix_and_budget():
    rows = [{} for _ in range(614)]
    assert prompt_stream(rows,80,4,2027) == prompt_stream(rows,200,4,2027)[:80]
    config = resolved_config(Path.cwd())
    assert config["student"]["updates"] == 80
    assert config["student"]["milestones"] == [0,10,40,80]
    assert config["cafd"]["teacher_rollouts_per_prompt_by_phase"] == [6,4,4,2,2]
    assert config["final"]["official_held_out_evaluations"] == 0


def test_update_telemetry_handles_in_memory_rollouts_without_update(monkeypatch, tmp_path):
    import json
    from cafd import mistral_anchored_v14 as entry, anchored_target
    def fake_update(*args):
        anchored_target.ACTIVE_STATS.add(torch.tensor([[0.5]]),torch.tensor([[1.]]),torch.tensor([[1.]]))
        return 0.1,2,0.3
    monkeypatch.setattr(entry, "_original_update", fake_update)
    monkeypatch.setattr(entry, "_current_update_index", 39)
    result = entry.measured_update("cafd",None,None,None,[{"row_id":"x"}],None,None,0,64,tmp_path)
    row = json.loads((tmp_path/"alpha_metrics.jsonl").read_text())
    assert row["update"] == 40
    assert row["alpha_mean"] == 0.5
    assert row["positions"] == 1
    assert anchored_target.ACTIVE_STATS is None
