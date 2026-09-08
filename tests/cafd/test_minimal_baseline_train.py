"""CPU fake-model integration tests for the isolated CAFD-MPC runner."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import cafd.minimal_baseline_train as runner
from cafd.minimal_baseline_objective import route_training_groups, absolute_teacher_coefficients
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
    cfg = runner.configuration("smoke","absolute_kd")
    cfg.update(token_chunk=2, rollouts_per_prompt=2, grad_clip=1., control_rollouts=2)
    monkeypatch.setattr(runner, "generate", fake_generate)
    return student, behavior, reference, pool, optimizer, cfg


def assert_state_same(model, snapshot):
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, snapshot[key], atol=0, rtol=0)



@pytest.mark.parametrize("condition", ["grpo", "absolute_kd"])
def test_four_routing_classes_use_verified_success_and_strict_variation(condition):
    rewards=torch.tensor([[0.]*8,[.1,.2]*4,[.1,1.]*4,[1.]*8],dtype=torch.float64)
    routing=route_training_groups(rewards,rewards==1.,condition=condition)
    assert routing.distill.tolist()==([True,True,False,False] if condition=="absolute_kd" else [False]*4)
    assert routing.rl.tolist()==[False,True,True,False]
    assert routing.skip.tolist()==([False,False,False,True] if condition=="absolute_kd" else [True,False,False,True])
    torch.testing.assert_close(routing.advantages[1].sum(),torch.tensor(0.),atol=2e-6,rtol=0)
    near=torch.tensor([[.1,.1+1e-14]],dtype=torch.float64)
    assert route_training_groups(near,near==1.,condition=condition).rl.item()
    with pytest.raises(ValueError,match="equivalent"):
        route_training_groups(rewards,torch.ones_like(rewards,dtype=torch.bool),condition=condition)


@pytest.mark.parametrize("u,expected",[(0.,{"r0":1.}),(.5,{"r0":.5,"r1":.5}),
    (1.,{"r1":1.}),(1.25,{"r1":.75,"r2":.25}),(2.,{"r2":1.})])
def test_absolute_coefficients_are_convex_route_interpolation(u,expected):
    assert absolute_teacher_coefficients(u,["r0","r1","r2"])==expected


def test_absolute_duplicate_checkpoint_is_coalesced_and_invalid_u_rejected():
    assert absolute_teacher_coefficients(.5,["r0","r0","r2"])=={"r0":1.}
    for u in [-1.,3.,float("nan")]:
        with pytest.raises(ValueError):
            absolute_teacher_coefficients(u,["r0","r1","r2"])


@pytest.mark.parametrize("condition",["grpo","absolute_kd"])
def test_round_matches_independent_dense_loss_gradient_and_one_clipped_update(setup,monkeypatch,condition):
    student,behavior,reference,pool,optimizer,cfg=setup
    cfg.update(runner.configuration("smoke",condition))
    cfg.update(token_chunk=2,rollouts_per_prompt=8,grad_clip=1.)
    rows=[
        make_row("failed-equal",[0.,0.],[[3,4],[4,2]]),
        make_row("failed-variable",[.1,.4],[[5],[6,7,8]]),
        make_row("mixed",[.1,1.],[[5],[6,7,8]]),
        make_row("all-success",[1.,1.],[[9,2],[10,3,2]]),
    ]
    expected=copy.deepcopy(student)
    desired_optimizer=FP32AdamW(expected.parameters(),lr=1e-3,weight_decay=0.)
    groups=[fake_generate(behavior,TOKENIZER,row,count=8,seed=0) for row in rows]
    z=sum(len(record["completion_ids"]) for group in groups for record in group)
    assert z==68
    expected_kd=0.;expected_rl=0.
    expected_loss=torch.tensor(0.)
    for group in groups:
        rewards=torch.tensor([r["reward"] for r in group],dtype=torch.float64)
        centered=rewards-rewards.mean()
        advantage=(centered/(centered.square().mean().sqrt()+1e-6)).float()
        use_kd=condition=="absolute_kd" and not any(r["full_pass"] for r in group)
        use_rl=max(r["reward"] for r in group)>min(r["reward"] for r in group)
        if not (use_kd or use_rl):
            continue
        for i,record in enumerate(group):
            ids,attention,mask=sequence_inputs(record,TOKENIZER,DEVICE)
            logp=expected.lm_head(expected(ids,attention)[:,:-1][mask]).log_softmax(-1)
            if use_kd:
                with torch.no_grad():
                    left=pool.models["r1"];right=pool.models["r2"]
                    logits=.75*left.lm_head(left(ids,attention)[:,:-1][mask])+.25*right.lm_head(right(ids,attention)[:,:-1][mask])
                    logq=logits.log_softmax(-1)
                term=(logq.exp()*(logq-logp)).sum()/z
                expected_loss=expected_loss+term
                expected_kd+=float(term.detach())
            if use_rl:
                current=logp.gather(-1,ids[:,1:][mask][:,None]).flatten()
                ratio=(current-torch.tensor(record["old_log_probs"])).exp()
                term=-torch.minimum(ratio*advantage[i],ratio.clamp(.8,1.2)*advantage[i]).sum()/z
                expected_loss=expected_loss+term
                expected_rl+=float(term.detach())
    expected_loss.backward()
    expected_gradients=[p.grad.clone() for p in expected.parameters()]
    desired_norm=torch.nn.utils.clip_grad_norm_(expected.parameters(),1.)
    desired_optimizer.step()
    initial_frozen=[copy.deepcopy(model.state_dict()) for model in [reference,*pool.models.values()]]
    def forbidden(*args,**kwargs):
        raise AssertionError("training must not score S0 or unrelated teachers")
    monkeypatch.setattr(reference,"forward",forbidden)
    monkeypatch.setattr(pool.models["r0"],"forward",forbidden)
    if condition=="grpo":
        monkeypatch.setattr(pool,"select",forbidden)
        monkeypatch.setattr(runner,"absolute_frozen_sources",forbidden)
        monkeypatch.setattr(runner,"exact_linear_forward_kl",forbidden)
    original_step=optimizer.step
    original_clip=torch.nn.utils.clip_grad_norm_
    steps=[];clips=[]
    def step(*args,**kwargs):
        steps.append(1)
        return original_step(*args,**kwargs)
    def clip(parameters,*args,**kwargs):
        parameters=list(parameters)
        for actual,expected_grad in zip(parameters,expected_gradients):
            torch.testing.assert_close(actual.grad,expected_grad,atol=3e-6,rtol=3e-6)
        clips.append(1)
        return original_clip(parameters,*args,**kwargs)
    monkeypatch.setattr(optimizer,"step",step)
    monkeypatch.setattr(torch.nn.utils,"clip_grad_norm_",clip)
    costs={}
    log,actual_groups=runner.train_round(student,behavior,reference,pool,["r0","r1","r2"],
        optimizer,TOKENIZER,rows,cfg,0,1.25,costs,DEVICE)
    assert len(steps)==len(clips)==1 and log["optimizer_step"]
    assert log["normalization_tokens"]==68
    assert log["eligible_tokens"]==(48 if condition=="absolute_kd" else 32)
    assert log["kd_tokens"]==(32 if condition=="absolute_kd" else 0)
    assert log["rl_tokens"]==32 and log["overlap_tokens"]==(16 if condition=="absolute_kd" else 0)
    assert log["kd_loss"]==pytest.approx(expected_kd,abs=2e-6)
    assert log["rl_loss"]==pytest.approx(expected_rl,abs=2e-6)
    assert log["gradient_norm"]==pytest.approx(float(desired_norm),abs=3e-6)
    assert costs["training_generated_tokens"]==68 and costs["verifier_calls"]==32
    if condition=="absolute_kd":
        assert costs["training_teacher_scored_tokens"]==64
        assert costs["training_reference_scored_tokens"]==0
        assert pool.calls==[{"r1","r2"}]
    else:
        assert "training_teacher_scored_tokens" not in costs
    for actual,desired in zip(student.parameters(),expected.parameters()):
        torch.testing.assert_close(actual,desired,atol=2e-6,rtol=2e-6)
    for model,state in zip([reference,*pool.models.values()],initial_frozen):
        assert_state_same(model,state)
        assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("condition,rewards",[("grpo",[0.,0.]),("grpo",[1.,1.]),("absolute_kd",[1.,1.])])
def test_legal_skip_preserves_existing_momentum_and_weight_decay(setup,monkeypatch,condition,rewards):
    student,behavior,reference,pool,optimizer,cfg=setup
    cfg.update(runner.configuration("smoke",condition))
    optimizer.param_groups[0]["weight_decay"]=.3
    for parameter in student.parameters():
        parameter.grad=torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    old_state=copy.deepcopy(student.state_dict())
    old_optimizer=copy.deepcopy(optimizer.state_dict())
    def forbidden(*args,**kwargs):
        raise AssertionError("all-skip round must not optimize or score KD")
    monkeypatch.setattr(optimizer,"step",forbidden)
    monkeypatch.setattr(torch.nn.utils,"clip_grad_norm_",forbidden)
    monkeypatch.setattr(pool,"select",forbidden)
    monkeypatch.setattr(runner,"absolute_frozen_sources",forbidden)
    log,_=runner.train_round(student,behavior,reference,pool,["r0","r1","r2"],
        optimizer,TOKENIZER,[make_row("skip",rewards,[[3,2],[4,5,2]])],cfg,0,.5,{},DEVICE)
    assert not log["optimizer_step"] and log["eligible_tokens"]==0
    assert log["normalization_tokens"]==5 and log["skip_reason"]=="no_eligible_loss"
    assert_state_same(student,old_state)
    after=optimizer.state_dict()
    for index,values in old_optimizer["state"].items():
        for key,value in values.items():
            if isinstance(value,torch.Tensor):
                torch.testing.assert_close(after["state"][index][key],value,atol=0,rtol=0)
            else:
                assert after["state"][index][key]==value


@pytest.mark.parametrize("condition",["grpo","absolute_kd"])
def test_budget_completion_requires_rounds_and_actual_eligibility_not_one_update_per_round(setup,condition):
    student,behavior,reference,pool,optimizer,cfg=setup
    cfg.update(runner.configuration("smoke",condition))
    cfg["prompts_per_round"]=1
    curve=[]
    for i,rewards in enumerate(([.1,1.],[1.,1.])):
        log,_=runner.train_round(student,behavior,reference,pool,["r0","r1","r2"],
            optimizer,TOKENIZER,[make_row("budget",rewards,[[3,2],[4,2]])],cfg,i,.5,{},DEVICE)
        curve.append(log)
    result=runner.validate_training_budget(curve,cfg,1)
    assert result["rounds"]==2 and result["actual_updates"]==1 and result["skipped_rounds"]==[2]
    with pytest.raises(RuntimeError,match="counter"):
        runner.validate_training_budget(curve,cfg,2)
    with pytest.raises(RuntimeError,match="exactly once"):
        runner.validate_training_budget(curve[:1],cfg,1)
    broken=copy.deepcopy(curve);broken[0]["optimizer_step"]=False
    with pytest.raises(RuntimeError,match="eligible loss"):
        runner.validate_training_budget(broken,cfg,0)
    broken=copy.deepcopy(curve);broken[1]["group_routing"][0]["use_rl"]=True
    with pytest.raises(RuntimeError,match="eligibility"):
        runner.validate_training_budget(broken,cfg,1)


@pytest.mark.parametrize("condition",["grpo","absolute_kd"])
def test_zero_update_budget_is_valid_when_all_groups_have_no_training_loss(setup,condition):
    student,behavior,reference,pool,optimizer,cfg=setup
    cfg.update(runner.configuration("smoke",condition))
    cfg["prompts_per_round"]=1
    curve=[]
    for i in range(2):
        log,_=runner.train_round(student,behavior,reference,pool,["r0","r1","r2"],
            optimizer,TOKENIZER,[make_row("skip",[1.,1.],[[3,2],[4,2]])],cfg,i,.5,{},DEVICE)
        curve.append(log)
    assert runner.validate_training_budget(curve,cfg,0)["actual_updates"]==0


def test_condition_configs_preserve_legacy_execution_protocol():
    from cafd.kd_retention_train import configuration as original_configuration
    allowed_changed={"method","target","kd_retention"}
    for profile in ["smoke","formal"]:
        old=original_configuration(profile)
        for condition in ["grpo","absolute_kd"]:
            cfg=runner.configuration(profile,condition)
            assert {k:(old[k],cfg[k]) for k in old if old[k]!=cfg[k]}.keys()<=allowed_changed
            assert cfg["total_rounds"]==(2 if profile=="smoke" else 200)
            assert cfg["maximum_gpus"]==1 and cfg["reserved_memory_gib"]==192
            assert cfg["diagnostic_teacher_overhead"] and cfg["diagnostic_only_control"]
            assert cfg["diagnostic_target"]==old["target"]
            assert cfg["lambda_rl"]==1. and cfg["lambda_kd"]==(1. if condition=="absolute_kd" else 0.)
    with pytest.raises(ValueError):
        runner.configuration("formal","unknown")


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
