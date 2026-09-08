import copy
from types import SimpleNamespace

import pytest
import torch

from cafd.fixed_path_kd import (target_coefficients,target_logits,train_update,
                               validate_config,validate_protocol_allocation)
from cafd.target_path_prepare import kd_config
from cafd.mpc_runtime import sequence_inputs
from cafd.optimizer import FP32AdamW


@pytest.mark.parametrize("u",[0.,.25,1.,1.999,2.,3.])
@pytest.mark.parametrize("kind",["cumulative_relative","absolute_teacher"])
def test_target_matches_exact_dense_formula_and_detaches(u,kind):
    torch.manual_seed(1)
    route=["t0","t1","t2","t3"]
    teacher={p:torch.randn(7,11,requires_grad=True) for p in route}
    s0=torch.randn(7,11,requires_grad=True)
    m=min(int(u),2)
    alpha=u-m
    expected=(1-alpha)*teacher[route[m]]+alpha*teacher[route[m+1]]
    if kind=="cumulative_relative":
        expected=expected+s0-teacher[route[0]]
    value=target_logits(kind,u,route,s0,teacher)
    torch.testing.assert_close(value,expected)
    assert not value.requires_grad


def test_zero_coefficients_are_removed_and_integer_cost_is_exact():
    route=["t0","t1","t2"]
    assert target_coefficients("cumulative_relative",0,route)==(1.,{})
    assert target_coefficients("cumulative_relative",1,route)==(1.,{"t0":-1.,"t1":1.})
    assert target_coefficients("absolute_teacher",1,route)==(0.,{"t1":1.})
    assert target_coefficients("absolute_teacher",2,route)==(0.,{"t2":1.})


@pytest.mark.parametrize("u",[-1.,float("nan"),float("inf"),3.1])
def test_invalid_progress_fails(u):
    with pytest.raises(ValueError):
        target_coefficients("absolute_teacher",u,["t0","t1","t2","t3"])


def configs(tmp_path):
    fit=[{"id":f"p{i}"} for i in range(20)]
    dev=[{"id":"dev"}]
    route=[{"checkpoint":f"t{i}"} for i in range(6)]
    return [kd_config(tmp_path,tmp_path/"newrun",kind,route,fit,dev)
            for kind in ("cumulative_relative","absolute_teacher")]


def test_two_configs_share_exact_schedule_and_prompt_stream(tmp_path):
    relative,absolute=configs(tmp_path)
    assert validate_config(relative) and validate_config(absolute)
    only={"target","reference","output_directory"}
    assert {k:v for k,v in relative.items() if k not in only}=={k:v for k,v in absolute.items() if k not in only}
    assert relative["u_by_update"]==sum(([u]*40 for u in [1.,2.,3.,4.,5.]),[])
    assert len(relative["prompt_stream"])==200


@pytest.mark.parametrize("key,value",[("rl_loss",True),("mpc",True),("top_p",.95),
    ("learning_rate",1e-5),("support","teacher_replay"),("frozen_test_accessed",True)])
def test_forbidden_config_drift_rejected(tmp_path,key,value):
    cfg=configs(tmp_path)[0]
    cfg[key]=value
    with pytest.raises(ValueError):
        validate_config(cfg)


class Tiny(torch.nn.Module):
    def __init__(self,hidden):
        super().__init__()
        self.embedding=torch.nn.Embedding(11,hidden)
        self.lm_head=torch.nn.Linear(hidden,11)
    def forward(self,ids,attention):
        return self.embedding(ids)


class Pool:
    def __init__(self):
        self.models={f"t{i}":Tiny(5+i).eval().requires_grad_(False) for i in range(3)}
        self.calls=[]
    def select(self,terms):
        self.calls.append(set(terms))
        return {p:self.models[p] for p in terms}


class Costs(dict):
    def add(self,key,value):
        self[key]=self.get(key,0)+value


@pytest.mark.parametrize("kind",["cumulative_relative","absolute_teacher"])
def test_pure_kd_trains_all_rewards_and_matches_full_vocab_dense(monkeypatch,kind):
    import cafd.mpc_runtime as runtime
    torch.manual_seed(511)
    device=torch.device("cpu")
    tokenizer=SimpleNamespace(eos_token_id=2,pad_token_id=0)
    student=Tiny(4)
    initial=copy.deepcopy(student).eval().requires_grad_(False)
    reference=initial if kind=="cumulative_relative" else None
    pool=Pool()
    expected=copy.deepcopy(student)
    optimizer=FP32AdamW(student.parameters(),lr=1e-3)
    expected_optimizer=FP32AdamW(expected.parameters(),lr=1e-3)
    rows=[dict(id="fail",reward=0.),dict(id="partial",reward=.3),dict(id="success",reward=1.)]
    def generate(model,tokenizer,row,*,count,**kwargs):
        return [dict(row_id=row["id"],prompt_ids=[1,7],completion_ids=([4,2] if j==0 else [8,9,2]),
                     reward=row["reward"],full_pass=row["reward"]==1.) for j in range(count)]
    monkeypatch.setattr(runtime,"generate",generate)
    cfg=dict(target=kind,route=["t0","t1","t2"],u_by_update=[1.25],
        rollouts_per_prompt=2,max_new_tokens=2048,max_prompt_tokens=4096,seed=2027,token_chunk=2,grad_clip=1.)
    records=[r for row in rows for r in generate(student,tokenizer,row,count=2)]
    z=sum(len(r["completion_ids"]) for r in records)
    dense_loss=0.
    for record in records:
        ids,attention,mask=sequence_inputs(record,tokenizer,device)
        h=expected(ids,attention)[:,:-1][mask]
        logp=expected.lm_head(h).log_softmax(-1)
        with torch.no_grad():
            t={p:model.lm_head(model(ids,attention)[:,:-1][mask]) for p,model in pool.models.items()}
            s0=initial.lm_head(initial(ids,attention)[:,:-1][mask])
            logq=target_logits(kind,1.25,cfg["route"],s0,t).log_softmax(-1)
        dense_loss=dense_loss+(logq.exp()*(logq-logp)).sum()/z
    dense_loss.backward()
    torch.nn.utils.clip_grad_norm_(expected.parameters(),1.)
    expected_optimizer.step()
    costs=Costs()
    log,actual=train_update(student,reference,pool,optimizer,tokenizer,rows,cfg,0,costs,device)
    assert log["tokens"]==15 and len(actual)==6
    assert log["loss"]==pytest.approx(float(dense_loss.detach()),rel=1e-5,abs=1e-6)
    for actual_parameter,desired in zip(student.parameters(),expected.parameters()):
        torch.testing.assert_close(actual_parameter,desired,atol=2e-6,rtol=2e-6)
    teacher_count=3 if kind=="cumulative_relative" else 2
    assert costs["training_teacher_scored_tokens"]==15*teacher_count
    assert costs["training_reference_scored_tokens"]==(15 if reference is not None else 0)
    assert all(p.grad is None for model in [initial,*pool.models.values()] for p in model.parameters())
    assert len(pool.calls)==1


@pytest.mark.parametrize("field",["SLURM_GPUS","SLURM_GPUS_ON_NODE","SLURM_JOB_GPUS"])
@pytest.mark.parametrize("count",[2,3,4])
def test_fixed_kd_enforces_two_gpus_after_shared_four_gpu_validator(field,count):
    from cafd.mpc_allocation import validate_allocation
    environ={"SLURM_JOB_ID":"123456","SLURM_JOB_NUM_NODES":"1","SLURM_MEM_PER_NODE":"192G"}
    environ[field]=(",".join(str(i) for i in range(count)) if field=="SLURM_JOB_GPUS" else str(count))
    allocation=validate_allocation(environ,"NVIDIA B200",visible_gpu_count=1)
    assert allocation["slurm_gpu_count_metadata"][field]==count
    assert allocation["maximum_cafd_gpus"]==4
    if count==2:
        validated=validate_protocol_allocation(allocation,{"maximum_gpus":2})
        assert validated["maximum_fixed_kd_gpus"]==2
    else:
        with pytest.raises(RuntimeError,match="exceeds frozen fixed-KD maximum_gpus=2"):
            validate_protocol_allocation(allocation,{"maximum_gpus":2})


@pytest.mark.parametrize("larger_field",["SLURM_GPUS_ON_NODE","SLURM_JOB_GPUS"])
def test_any_larger_actual_field_cannot_be_hidden_by_two_gpu_field(larger_field):
    allocation={"slurm_gpu_count_metadata":{"SLURM_GPUS":2,larger_field:4}}
    with pytest.raises(RuntimeError,match=larger_field):
        validate_protocol_allocation(allocation,{"maximum_gpus":2})


def test_missing_actual_reservation_metadata_fails_closed():
    with pytest.raises(RuntimeError,match="metadata is missing"):
        validate_protocol_allocation({"slurm_gpu_count_metadata":{}},{"maximum_gpus":2})


def test_protocol_limit_is_read_from_frozen_config_not_literal_four_or_two():
    with pytest.raises(RuntimeError,match="maximum_gpus=1"):
        validate_protocol_allocation({"slurm_gpu_count_metadata":{"SLURM_GPUS":2}}, {"maximum_gpus":1})
