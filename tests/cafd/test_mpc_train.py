"""CPU fake-model integration tests for the isolated CAFD-MPC runner."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import cafd.mpc_train as runner
from cafd.mpc_objective import route_reward_groups, teacher_coefficients
from cafd.mpc_runtime import independent_copy, sequence_inputs
from cafd.optimizer import FP32AdamW

DEVICE = torch.device("cpu")
TOKENIZER = SimpleNamespace(eos_token_id=2, pad_token_id=0)


class TinyHidden(torch.nn.Module):
    def __init__(self, width=4, vocab=11):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, width)
        self.lm_head = torch.nn.Linear(width, vocab, bias=True)

    def forward(self, ids, attention):
        return self.embedding(ids)


class FakePool:
    def __init__(self, models):
        self.models = models
        self.calls = []
        self.loads = 0

    def select(self, coefficients):
        keys = set(coefficients)
        self.calls.append(keys)
        return {key: self.models[key] for key in keys}


def make_row(identifier, rewards, completions, family="contains_count"):
    return {"id": identifier, "problem_family": family, "ground_truth": [],
            "rewards": rewards, "completions": completions}


def fake_generate(model, tokenizer, row, *, count, seed, **kwargs):
    records = []
    for slot in range(count):
        tokens = list(row["completions"][slot % len(row["completions"])])
        reward = row["rewards"][slot % len(row["rewards"])]
        record = dict(row_id=row["id"], family=row["problem_family"],
                      prompt_ids=[1, 7], completion_ids=tokens, reward=reward,
                      full_pass=reward == 1., tier="full_pass" if reward == 1 else "partial",
                      length_cap=False, source="current_student", seed=seed)
        ids, attention, mask = sequence_inputs(record, tokenizer, DEVICE)
        with torch.no_grad():
            hidden = model(ids, attention)[:, :-1][mask]
            logp = F.linear(hidden.float(), model.lm_head.weight.float(),
                            model.lm_head.bias.float()).log_softmax(-1)
            record["old_log_probs"] = logp.gather(-1, ids[:, 1:][mask][:, None]).flatten().tolist()
        records.append(record)
    return records


@pytest.fixture
def setup(monkeypatch):
    torch.manual_seed(892)
    student = TinyHidden()
    reference = independent_copy(student)
    behavior = independent_copy(student)
    teachers = {f"r{i}": TinyHidden(width=5 + i).eval().requires_grad_(False) for i in range(3)}
    pool = FakePool(teachers)
    optimizer = FP32AdamW(student.parameters(), lr=1e-3, weight_decay=0.)
    cfg = runner.configuration("smoke")
    cfg.update(token_chunk=2, rollouts_per_prompt=2, grad_clip=1., control_rollouts=2)
    monkeypatch.setattr(runner, "generate", fake_generate)
    return student, behavior, reference, pool, optimizer, cfg


def assert_state_same(model, snapshot):
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, snapshot[key], atol=0, rtol=0)


def test_mixed_round_matches_dense_update_and_scores_teachers_only_for_distill(setup):
    student, behavior, reference, pool, optimizer, cfg = setup
    rows = [
        make_row("d", [0., 0.], [[3, 4], [4, 2]]),
        make_row("r", [.1, .4], [[5], [6, 7, 8]]),
        make_row("skip", [1., 1.], [[9, 2], [10, 3, 2]]),
    ]
    start = copy.deepcopy(student.state_dict())
    frozen_states = [copy.deepcopy(reference.state_dict())] + [
        copy.deepcopy(model.state_dict()) for model in pool.models.values()]
    expected = copy.deepcopy(student)
    expected_optimizer = FP32AdamW(expected.parameters(), lr=1e-3, weight_decay=0.)
    groups = [fake_generate(behavior, TOKENIZER, row, count=2, seed=0) for row in rows]
    z = sum(len(record["completion_ids"]) for group in groups for record in group)
    assert z == 13
    routing = route_reward_groups(torch.tensor([row["rewards"] for row in rows]))
    coefficients = teacher_coefficients(1.25, ["r0", "r1", "r2"])
    expected_loss = 0.
    for j, group in enumerate(groups):
        if routing.skip[j]:
            continue
        for l, record in enumerate(group):
            ids, attention, mask = sequence_inputs(record, TOKENIZER, DEVICE)
            hidden = expected(ids, attention)[:, :-1][mask]
            logp = expected.lm_head(hidden).log_softmax(-1)
            if routing.distill[j]:
                with torch.no_grad():
                    target = reference.lm_head(reference(ids, attention)[:, :-1][mask])
                    for key, coefficient in coefficients.items():
                        model = pool.models[key]
                        target = target + coefficient * model.lm_head(model(ids, attention)[:, :-1][mask])
                    logq = target.log_softmax(-1)
                loss = (logq.exp() * (logq - logp)).sum() / z
            else:
                labels = ids[:, 1:][mask]
                selected = logp.gather(-1, labels[:, None]).flatten()
                old = torch.tensor(record["old_log_probs"])
                ratio = (selected - old).exp()
                advantage = routing.advantages[j, l]
                loss = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).sum() / z
            expected_loss = expected_loss + loss
    expected_loss.backward()
    torch.nn.utils.clip_grad_norm_(expected.parameters(), cfg["grad_clip"])
    expected_optimizer.step()

    costs = {}
    log, generated = runner.train_round(student, behavior, reference, pool, ["r0", "r1", "r2"],
        optimizer, TOKENIZER, rows, cfg, 0, 1.25, costs, DEVICE)
    assert log["normalization_tokens"] == 13
    assert (log["distill_groups"], log["rl_groups"], log["skip_groups"]) == (1, 1, 1)
    assert log["optimizer_step"] and log["gradient_norm"] > 0
    assert log["max_behavior_logprob_delta"] < 2e-6
    assert costs["training_generated_tokens"] == 13
    assert costs["training_teacher_scored_tokens"] == 4 * 3
    assert costs["training_reference_scored_tokens"] == 4
    assert costs["verifier_calls"] == 6
    assert pool.calls == [{"r0", "r1", "r2"}]
    assert [record["row_id"] for group in generated for record in group] == ["d", "d", "r", "r", "skip", "skip"]
    for actual, desired in zip(student.parameters(), expected.parameters()):
        torch.testing.assert_close(actual, desired, atol=2e-6, rtol=2e-6)
    assert_state_same(behavior, start)
    for model, snapshot in zip([reference, *pool.models.values()], frozen_states):
        assert_state_same(model, snapshot)
        assert all(parameter.grad is None for parameter in model.parameters())


def test_all_success_skips_optimizer_momentum_weight_decay_and_teacher(setup):
    student, behavior, reference, pool, optimizer, cfg = setup
    optimizer.param_groups[0]["weight_decay"] = .3
    for parameter in student.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()  # Nonzero preexisting moments must not move on a skipped round.
    optimizer.zero_grad(set_to_none=True)
    before = copy.deepcopy(student.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    costs = {}
    rows = [make_row("all-success", [1., 1.], [[3, 2], [4, 5, 2]])]
    log, _ = runner.train_round(student, behavior, reference, pool, ["r0", "r1", "r2"],
        optimizer, TOKENIZER, rows, cfg, 1, 1.25, costs, DEVICE)
    assert not log["optimizer_step"]
    assert log["normalization_tokens"] == 5 and log["skip_groups"] == 1
    assert_state_same(student, before)
    assert costs["training_generated_tokens"] == 5 and costs["verifier_calls"] == 2
    assert "training_teacher_scored_tokens" not in costs and pool.calls == []
    after = optimizer.state_dict()
    for index, state in optimizer_before["state"].items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(after["state"][index][key], value, atol=0, rtol=0)
            else:
                assert after["state"][index][key] == value


def test_all_failed_but_variable_partial_reward_uses_only_rl(setup):
    student, behavior, reference, pool, optimizer, cfg = setup
    rows = [make_row("partial", [.1, .5], [[3], [4, 2]])]
    costs = {}
    log, _ = runner.train_round(student, behavior, reference, pool, ["r0", "r1", "r2"],
        optimizer, TOKENIZER, rows, cfg, 1, .5, costs, DEVICE)
    assert log["rl_groups"] == 1 and log["distill_groups"] == 0
    assert log["optimizer_step"] and pool.calls == []
    assert "training_teacher_scored_tokens" not in costs
    assert all(parameter.grad is None for parameter in reference.parameters())


def test_fixed_full_vocabulary_endpoint_cache_is_reused_and_observation_has_no_grad(setup, tmp_path):
    student, behavior, reference, pool, optimizer, cfg = setup
    families = ["contains_count", "contains_ordered", "contains_substring"]
    controls = [make_row(f"c-{i}", [.1 * i, 1.], [[3, 2], [4, 2]], family=family)
                for i, family in enumerate(families)]
    costs = {}
    cfg["control_rollouts"] = 2
    cache = runner.build_probe_cache(tmp_path, controls, reference, pool, ["r0", "r1", "r2"],
                                    TOKENIZER, cfg, costs, DEVICE)
    assert pool.calls == [{"r0", "r2"}], "Final cache uses endpoint minus T0, not intermediate target"
    assert costs["probe_setup_teacher_scored_tokens"] == 12
    assert cache["full_vocabulary"] and cache["dtype"] == "float32"
    before_costs = copy.deepcopy(costs)
    reloaded = runner.build_probe_cache(tmp_path, controls, reference, pool, ["r0", "r1", "r2"],
                                       TOKENIZER, cfg, costs, DEVICE)
    assert reloaded == cache and costs == before_costs
    assert len(pool.calls) == 1
    states = runner.observe_once(student, TOKENIZER, controls, cache, families, tmp_path,
                                 cfg, 0, costs, DEVICE)
    assert states.shape == (3, 3) and np.isfinite(states).all()
    np.testing.assert_allclose(states[:, 0], [.5, .55, .6], atol=1e-7)
    np.testing.assert_allclose(states[:, 1], [.5, .5, .5], atol=0)
    assert ((states[:, 2] >= 0) & (states[:, 2] < 1)).all()
    assert costs["control_generated_tokens"] == 12
    assert costs["control_student_scored_tokens"] == 6
    assert all(parameter.grad is None for parameter in student.parameters())
    assert all(parameter.grad is None for parameter in reference.parameters())


def test_smoke_profile_is_small_and_explicitly_not_formal():
    smoke, formal = runner.configuration("smoke"), runner.configuration("formal")
    assert smoke["not_a_formal_result"] is True and smoke["milestones"] == []
    assert smoke["total_rounds"] == 8 and smoke["max_new_tokens"] == 64
    assert formal["total_rounds"] == 200 and formal["max_new_tokens"] == 2048
    assert formal["prompts_per_round"] * formal["rollouts_per_prompt"] == 32
    assert formal["block_rounds"] == 10 and formal["horizon"] == 2
    assert formal["maximum_gpus"] <= 2 and formal["reserved_memory_gib"] <= 192
    assert not smoke["frozen_test_evaluated"] and not formal["frozen_test_evaluated"]
