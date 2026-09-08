from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from cafd.mpc_objective import (
    clipped_rl_loss, completion_prediction_mask, cumulative_target_logits,
    cumulative_target_probs, exact_linear_forward_kl, exact_linear_token_log_probs,
    mixed_objective_reference, route_reward_groups, teacher_coefficients,
)


def test_target_endpoints_integer_boundaries_and_frozen_gradients():
    torch.manual_seed(2)
    route = ["T0", "T1", "T2", "T3"]
    s0 = torch.randn(4, 17, requires_grad=True)
    teachers = {name: torch.randn(4, 17, requires_grad=True) for name in route}
    torch.testing.assert_close(cumulative_target_logits(s0, {}, 0, route), s0.detach())
    torch.testing.assert_close(
        cumulative_target_logits(s0, teachers, 3, route),
        s0.detach() + teachers["T3"].detach() - teachers["T0"].detach(),
    )
    for boundary in (1.0, 2.0):
        left = cumulative_target_probs(s0, teachers, boundary - 1e-6, route)
        exact = cumulative_target_probs(s0, teachers, boundary, route)
        right = cumulative_target_probs(s0, teachers, boundary + 1e-6, route)
        torch.testing.assert_close(left, exact, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(right, exact, atol=2e-6, rtol=2e-6)
    q = cumulative_target_probs(s0, teachers, 1.4, route)
    assert q.dtype == torch.float32 and not q.requires_grad
    torch.testing.assert_close(q.sum(-1), torch.ones(4))
    assert s0.grad is None and all(t.grad is None for t in teachers.values())


def test_teacher_coefficients_deduplicate_and_drop_only_exact_zero():
    route = ["zero", "first", "second", "last"]
    assert teacher_coefficients(0, route) == {}
    assert teacher_coefficients(0.25, route) == {"zero": -0.25, "first": 0.25}
    assert teacher_coefficients(1, route) == {"zero": -1.0, "first": 1.0}
    assert teacher_coefficients(1.5, route) == {"zero": -1.0, "first": 0.5, "second": 0.5}
    assert teacher_coefficients(3, route) == {"zero": -1.0, "last": 1.0}
    assert teacher_coefficients(2, ["zero", "same", "same"]) == {"zero": -1.0, "same": 1.0}
    assert teacher_coefficients(1, ["zero", "zero"]) == {}
    assert len(teacher_coefficients(1e-14, route)) == 2


@pytest.mark.parametrize("u", [-0.1, 4.0, math.nan, math.inf])
def test_target_progress_rejects_invalid(u):
    with pytest.raises(ValueError):
        teacher_coefficients(u, ["T0", "T1", "T2", "T3"])


def test_routing_all_fail_partial_variation_success_and_population_std():
    rewards = torch.tensor([[0., 0.], [.1, .1], [.1, .4], [1., .5], [1., 1.]], requires_grad=True)
    routed = route_reward_groups(rewards, rewards.detach() == 1)
    assert routed.distill.tolist() == [True, True, False, False, False]
    assert routed.rl.tolist() == [False, False, True, True, False]
    assert routed.skip.tolist() == [False, False, False, False, True]
    expected = torch.tensor([-.15, .15]) / (.15 + 1e-6)
    torch.testing.assert_close(routed.advantages[2], expected)
    assert routed.should_step and not routed.advantages.requires_grad
    assert not route_reward_groups(torch.ones(2, 8)).should_step


@pytest.mark.parametrize("rewards", [
    [[math.nan, 0.]], [[0., math.inf]], [[-.1, 0.]], [[1.1, 0.]], [[0.]],
])
def test_routing_rejects_invalid_rewards(rewards):
    with pytest.raises(ValueError):
        route_reward_groups(torch.tensor(rewards))


def test_verifier_full_pass_consistency_is_mandatory():
    with pytest.raises(ValueError, match="equivalent"):
        route_reward_groups(torch.tensor([[.5, 1.]]), torch.tensor([[True, True]]))
    with pytest.raises(ValueError, match="boolean"):
        route_reward_groups(torch.tensor([[.5, 1.]]), torch.tensor([[0., .5]]))


def test_fp64_reward_routing_does_not_round_near_success_or_erase_variation():
    rewards = torch.tensor([[1. - 1e-10, 1. - 1e-10], [.1, .1 + 1e-10]], dtype=torch.float64)
    routed = route_reward_groups(rewards, torch.zeros(2, 2, dtype=torch.bool))
    assert routed.distill.tolist() == [True, False]
    assert routed.rl.tolist() == [False, True]
    assert not routed.skip.any()
    assert routed.advantages[1, 0] < 0 and routed.advantages[1, 1] > 0


def test_completion_mask_includes_first_completion_and_first_eos_only():
    ids = torch.tensor([[7, 8, 3, 2, 9, 0], [7, 8, 2, 0, 0, 0], [7, 8, 0, 0, 0, 0]])
    attention = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]])
    mask = completion_prediction_mask(ids, attention, torch.tensor([2, 2, 2]), eos_token_id=2)
    assert mask.tolist() == [
        [False, True, True, False, False],
        [False, True, False, False, False],
        [False, False, False, False, False],
    ]


def test_reference_shared_normalizer_includes_skip_tokens_and_detaches_targets():
    torch.manual_seed(3)
    student = torch.randn(3, 2, 4, 7, requires_grad=True)
    target = torch.randn_like(student, requires_grad=True)
    rewards = torch.tensor([[.1, .1], [.1, .4], [1., 1.]])
    mask = torch.tensor([
        [[1, 1, 0, 0], [1, 1, 1, 0]],
        [[1, 1, 1, 1], [1, 0, 0, 0]],
        [[1, 1, 1, 1], [1, 1, 0, 0]],
    ], dtype=torch.bool)
    ids = torch.randint(7, mask.shape)
    old = torch.log_softmax(student.detach(), -1).gather(-1, ids[..., None]).squeeze(-1).requires_grad_()
    result = mixed_objective_reference(student, target, old, rewards, mask, ids)
    assert result.normalization_tokens == 16
    logq = torch.log_softmax(target.detach()[0][mask[0]], -1)
    logp = torch.log_softmax(student[0][mask[0]], -1)
    expected_kl = (logq.exp() * (logq - logp)).sum() / 16
    torch.testing.assert_close(result.distillation_loss, expected_kl)
    assert result.should_step
    result.loss.backward()
    assert target.grad is None and old.grad is None
    assert torch.count_nonzero(student.grad[2]) == 0
    assert torch.count_nonzero(student.grad[~mask]) == 0


def test_reference_does_not_read_teacher_entries_for_rl_or_skip_groups():
    torch.manual_seed(5)
    student = torch.randn(3, 2, 2, 5, requires_grad=True)
    target = torch.full_like(student, float("nan"))
    target[0] = 0
    rewards = torch.tensor([[0., 0.], [.1, .2], [1., 1.]])
    mask = torch.ones(3, 2, 2, dtype=torch.bool)
    result = mixed_objective_reference(student, target, torch.zeros(3, 2, 2), rewards, mask, torch.zeros(3, 2, 2, dtype=torch.long))
    assert torch.isfinite(result.loss)


def test_all_success_and_empty_completions_signal_entire_optimizer_skip():
    student = torch.randn(2, 2, 3, 5, requires_grad=True)
    rewards = torch.ones(2, 2)
    shape = (2, 2, 3)
    for mask in (torch.ones(shape, dtype=torch.bool), torch.zeros(shape, dtype=torch.bool)):
        result = mixed_objective_reference(student, None, torch.zeros(shape), rewards, mask, torch.zeros(shape, dtype=torch.long))
        assert not result.should_step
        assert result.loss.item() == 0
    # Momentum and decay are preserved by respecting the explicit should_step.
    parameter = torch.nn.Parameter(torch.tensor(1.))
    opt = torch.optim.AdamW([parameter], lr=.1, weight_decay=.1)
    parameter.square().backward()
    opt.step()
    before = parameter.detach().clone()
    moment = opt.state[parameter]["exp_avg"].clone()
    if result.should_step:
        opt.step()
    torch.testing.assert_close(parameter, before)
    torch.testing.assert_close(opt.state[parameter]["exp_avg"], moment)


def test_clipped_rl_is_symmetric_and_old_probabilities_detached():
    ratio = torch.tensor([1.5, .5, 1.5, .5])
    current = ratio.log().requires_grad_()
    old = torch.zeros(4, requires_grad=True)
    advantages = torch.tensor([1., 1., -1., -1.], requires_grad=True)
    loss = clipped_rl_loss(current, old, advantages, torch.ones(4, dtype=torch.bool), normalization_tokens=4)
    expected = -torch.tensor([1.2, .5, -1.5, -.8]).sum() / 4
    torch.testing.assert_close(loss, expected)
    loss.backward()
    torch.testing.assert_close(current.grad, torch.tensor([0., -.5, 1.5, 0.]) / 4)
    assert old.grad is None and advantages.grad is None


def _linear_fixture(dtype=torch.float32):
    torch.manual_seed(29)
    h = torch.randn(11, 4, dtype=dtype, requires_grad=True)
    w = torch.randn(19, 4, dtype=dtype, requires_grad=True)
    b = torch.randn(19, dtype=dtype, requires_grad=True)
    sources = []
    for coefficient, width in ((1., 4), (-1., 6), (.7, 6), (.3, 6)):
        sh = torch.randn(11, width, dtype=dtype, requires_grad=True)
        sw = torch.randn(19, width, dtype=dtype, requires_grad=True)
        sb = torch.randn(19, dtype=dtype, requires_grad=True)
        sources.append((coefficient, sh, sw, sb))
    return h, w, b, sources


@pytest.mark.parametrize("chunk", [1, 3, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_streaming_kl_matches_dense_value_hidden_head_bias_and_upstream_gradients(chunk, dtype):
    h, w, b, sources = _linear_fixture(dtype)
    hd, wd, bd = [x.detach().clone().requires_grad_() for x in (h, w, b)]
    expected_target = sum(c * F.linear(sh.detach().float(), sw.detach().float(), sb.detach().float()) for c, sh, sw, sb in sources)
    logq = F.log_softmax(expected_target, -1)
    logp = F.log_softmax(F.linear(hd.float(), wd.float(), bd.float()), -1)
    expected = (logq.exp() * (logq - logp)).sum() / 37
    actual = exact_linear_forward_kl(h, w, sources, normalization_tokens=37, token_chunk=chunk, student_bias=b)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    tol = 3e-3 if dtype == torch.bfloat16 else 3e-6
    for result, reference in ((h, hd), (w, wd), (b, bd)):
        torch.testing.assert_close(result.grad, reference.grad, atol=tol, rtol=tol)
    for _, sh, sw, sb in sources:
        assert sh.grad is None and sw.grad is None and sb.grad is None


def test_streaming_kl_propagates_into_student_backbone_only():
    torch.manual_seed(6)
    backbone = torch.nn.Linear(3, 4)
    frozen = torch.nn.Linear(3, 5)
    inputs = torch.randn(7, 3)
    h = backbone(inputs)
    sh = frozen(inputs)
    head = torch.randn(11, 4, requires_grad=True)
    frozen_head = torch.randn(11, 5, requires_grad=True)
    loss = exact_linear_forward_kl(h, head, [(1., sh, frozen_head, None)], normalization_tokens=17, token_chunk=2)
    loss.backward()
    assert backbone.weight.grad is not None and torch.count_nonzero(backbone.weight.grad) > 0
    assert frozen.weight.grad is None and frozen_head.grad is None


def test_streaming_kl_hidden_gradient_matches_central_finite_difference():
    h, w, b, sources = _linear_fixture()
    loss = exact_linear_forward_kl(h, w, sources, normalization_tokens=37, token_chunk=3, student_bias=b)
    loss.backward()
    numerical = []
    for index in [(0, 1), (3, 2), (10, 3)]:
        plus, minus = h.detach().clone(), h.detach().clone()
        plus[index] += 1e-3
        minus[index] -= 1e-3
        fplus = exact_linear_forward_kl(plus, w.detach(), sources, normalization_tokens=37, token_chunk=3, student_bias=b.detach())
        fminus = exact_linear_forward_kl(minus, w.detach(), sources, normalization_tokens=37, token_chunk=3, student_bias=b.detach())
        numerical.append((fplus - fminus) / .002)
    reference = torch.stack([h.grad[i] for i in [(0, 1), (3, 2), (10, 3)]])
    torch.testing.assert_close(torch.stack(numerical), reference, atol=2e-4, rtol=1e-2)


@pytest.mark.parametrize("chunk", [1, 4, 50])
def test_streaming_sampled_log_probs_match_dense_and_gradients(chunk):
    h, w, b, _ = _linear_fixture()
    hd, wd, bd = [x.detach().clone().requires_grad_() for x in (h, w, b)]
    ids = torch.arange(11)
    weights = torch.linspace(-2, 2, 11)
    reference = F.log_softmax(F.linear(hd.float(), wd.float(), bd.float()), -1).gather(-1, ids[:, None]).squeeze(-1)
    actual = exact_linear_token_log_probs(h, w, ids, token_chunk=chunk, student_bias=b)
    torch.testing.assert_close(actual, reference)
    (actual * weights).sum().backward()
    (reference * weights).sum().backward()
    for result, expected in ((h, hd), (w, wd), (b, bd)):
        torch.testing.assert_close(result.grad, expected.grad, atol=3e-6, rtol=3e-6)


def test_streaming_does_not_save_completion_times_vocabulary_distributions():
    h, w, b, sources = _linear_fixture()
    saved_shapes = []
    def pack(tensor):
        saved_shapes.append(tuple(tensor.shape))
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = exact_linear_forward_kl(h, w, sources, normalization_tokens=37, token_chunk=3, student_bias=b)
        logp = exact_linear_token_log_probs(h, w, torch.arange(11), token_chunk=3, student_bias=b)
    assert (11, 19) not in saved_shapes and (3, 19) not in saved_shapes
    (loss + logp.sum()).backward()


def test_fp32_projection_contract_survives_outer_autocast():
    h, w, b, sources = _linear_fixture()
    ordinary = exact_linear_forward_kl(h, w, sources, normalization_tokens=37, student_bias=b)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast = exact_linear_forward_kl(h, w, sources, normalization_tokens=37, student_bias=b)
    assert autocast.dtype == torch.float32
    torch.testing.assert_close(autocast, ordinary, atol=1e-7, rtol=1e-7)


def test_streaming_empty_position_set_has_zero_student_gradients():
    hidden = torch.empty(0, 4, requires_grad=True)
    weight = torch.randn(9, 4, requires_grad=True)
    source_h = torch.empty(0, 6, requires_grad=True)
    source_w = torch.randn(9, 6, requires_grad=True)
    loss = exact_linear_forward_kl(hidden, weight, [(1., source_h, source_w, None)], normalization_tokens=5)
    assert loss.item() == 0
    loss.backward()
    assert weight.grad.count_nonzero() == 0 and source_w.grad is None
