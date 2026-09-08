"""Prepared fixed-path pure KD comparison. Validation is the CLI default.

Training requires an explicit --execute-training inside an existing B200 Slurm
allocation. This module never submits jobs and never imports a control predictor
or reward router. No test split is loaded.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from .mpc_objective import teacher_coefficients,exact_linear_forward_kl


def target_coefficients(target,u,route):
    """Return S0 multiplier and coalesced actual Teacher terms."""
    if target=="cumulative_relative":
        return 1.,teacher_coefficients(u,route)
    if target!="absolute_teacher":
        raise ValueError("exactly two predeclared target types are supported")
    if len(route)<2 or not math.isfinite(u) or not 0 <= u <= len(route)-1:
        raise ValueError("u outside the fixed Teacher route")
    m=min(math.floor(u),len(route)-2)
    alpha=u-m
    terms={}
    for path,c in ((route[m],1-alpha),(route[m+1],alpha)):
        terms[path]=terms.get(path,0.)+c
    return 0.,{p:c for p,c in terms.items() if c!=0.}


@torch.no_grad()
def target_logits(target,u,route,s0,teachers):
    reference,terms=target_coefficients(target,u,route)
    result=s0.detach().float()*reference if reference else None
    for path,c in terms.items():
        term=teachers[path].detach().float()
        result=c*term if result is None else result+c*term
    if result is None:
        raise ValueError("empty absolute target")
    return result


def validate_config(config):
    if config["protocol"]!="CAFD-fixed-target-pure-KD-v1":
        raise ValueError("wrong fixed KD protocol")
    if config["target"] not in {"cumulative_relative","absolute_teacher"}:
        raise ValueError("unknown target")
    expected=dict(seed=2027,total_updates=200,prompts_per_update=4,rollouts_per_prompt=8,
        max_new_tokens=2048,max_prompt_tokens=4096,temperature=1.,top_p=1.,top_k=0,
        sample=True,rl_loss=False,reward_routing=False,mpc=False,adaptive_schedule=False,
        support="current_student_only",loss="exact_full_vocabulary_forward_KL",
        normalization="all_valid_completion_tokens_across_32_slots",
        optimizer="FP32-master-AdamW",learning_rate=1e-6,betas=[.9,.999],eps=1e-8,
        weight_decay=0.,grad_clip=1.,lr_schedule="constant",maximum_gpus=2,runner_gpus=1,
        frozen_test_accessed=False,milestones=[0,10,40,80,120,160,200])
    if any(config.get(k)!=v for k,v in expected.items()):
        raise ValueError("frozen common KD hyperparameters changed")
    expected_reference="permanent_initial_S0" if config["target"]=="cumulative_relative" else None
    if config.get("reference")!=expected_reference:
        raise ValueError("target reference policy changed")
    m=len(config["route"])-1
    if m<1 or len(set(config["route"]))!=m+1:
        raise ValueError("Teacher route must contain distinct checkpoints")
    expected_u=[m*(segment+1)/5 for segment in range(5) for _ in range(40)]
    if config["u_by_update"] != expected_u or min(expected_u)<=0:
        raise ValueError("fixed nonzero five-stage schedule changed")
    stream=config["prompt_stream"]
    if len(stream)!=200 or len(config["training_ids"])!=len(set(config["training_ids"])):
        raise ValueError("invalid frozen prompt stream")
    for i,item in enumerate(stream):
        if any(type(j) is not int or not 0<=j<len(config["training_ids"]) for j in item["indices"]):
            raise ValueError("invalid prompt index")
        if item["update"]!=i+1 or len(item["indices"])!=4 or item["ids"]!=[config["training_ids"][j] for j in item["indices"]]:
            raise ValueError("prompt indices and IDs disagree")
    if set(config["training_ids"])&set(config["development_ids"]):
        raise ValueError("training/development overlap")
    for u in expected_u:
        target_coefficients(config["target"],u,config["route"])
    return True


def validate_protocol_allocation(allocation,config):
    """Enforce the frozen comparison limit on every actual Slurm GPU count.

    The shared allocation validator permits four GPUs for older CAFD protocols.
    This experiment has the stricter frozen limit of two. One visible GPU alone
    cannot establish the total reservation, so absent metadata fails closed.
    """
    maximum=config["maximum_gpus"]
    if type(maximum) is not int or maximum<1:
        raise RuntimeError("invalid frozen maximum_gpus")
    metadata=allocation["slurm_gpu_count_metadata"]
    if not metadata:
        raise RuntimeError("actual Slurm GPU reservation metadata is missing")
    for field,count in metadata.items():
        if type(count) is not int or count<1:
            raise RuntimeError(f"invalid validated GPU count {field}={count!r}")
        if count>maximum:
            raise RuntimeError(f"{field}={count} exceeds frozen fixed-KD maximum_gpus={maximum}")
    return dict(allocation,maximum_fixed_kd_gpus=maximum)


@torch.no_grad()
def frozen_target_sources(record,tokenizer,reference,teachers,target,u,route,device):
    from .mpc_runtime import sequence_inputs
    ref_c,terms=target_coefficients(target,u,route)
    if ref_c and reference is None:
        raise ValueError("relative target requires permanent S0")
    ids,attention,mask=sequence_inputs(record,tokenizer,device)
    models=([(ref_c,reference)] if ref_c else [])+[(c,teachers[p]) for p,c in terms.items()]
    sources=[]
    for coefficient,model in models:
        hidden=model(ids,attention)[:,:-1][mask]
        head=model.lm_head
        sources.append((coefficient,hidden.detach(),head.weight.detach(),None if head.bias is None else head.bias.detach()))
    return sources


def train_update(student,reference,pool,optimizer,tokenizer,rows,cfg,index,costs,device):
    from .mpc_runtime import generate,sequence_inputs
    from .data import rollout_seed
    records=[]
    for slot,row in enumerate(rows):
        records.extend(generate(student,tokenizer,row,count=cfg["rollouts_per_prompt"],
            max_new_tokens=cfg["max_new_tokens"],max_prompt_tokens=cfg["max_prompt_tokens"],
            seed=rollout_seed(index,slot,0,cfg["seed"])))
    tokens=sum(len(r["completion_ids"]) for r in records)
    if tokens<=0:
        raise RuntimeError("empty sampled batch; cannot silently drop a formal update")
    u=cfg["u_by_update"][index]
    ref_c,terms=target_coefficients(cfg["target"],u,cfg["route"])
    teachers=pool.select(terms)
    optimizer.zero_grad(set_to_none=True)
    student.train()
    loss_total=0.
    for r in records:
        ids,attention,mask=sequence_inputs(r,tokenizer,device)
        sources=frozen_target_sources(r,tokenizer,reference,teachers,cfg["target"],u,cfg["route"],device)
        hidden=student(ids,attention)[:,:-1][mask]
        head=student.lm_head
        loss=exact_linear_forward_kl(hidden,head.weight,sources,normalization_tokens=tokens,
            token_chunk=cfg["token_chunk"],student_bias=head.bias)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite exact KD loss")
        loss_total+=float(loss.detach())
        loss.backward()
        del hidden,sources,loss
    gradient=torch.nn.utils.clip_grad_norm_(student.parameters(),cfg["grad_clip"],error_if_nonfinite=True)
    optimizer.step()
    optimizer.assert_fp32_states()
    optimizer.zero_grad(set_to_none=True)
    if any(p.grad is not None for model in list(teachers.values())+([reference] if reference else []) for p in model.parameters()):
        raise RuntimeError("frozen target gradient leak")
    costs.add("training_generated_tokens",tokens)
    costs.add("training_teacher_scored_tokens",tokens*len(terms))
    costs.add("training_reference_scored_tokens",tokens*int(bool(ref_c)))
    costs.add("training_student_forward_backward_tokens",tokens)
    costs.add("training_verifier_calls",len(records))
    costs.add("training_prompt_tokens",sum(len(r["prompt_ids"]) for r in records))
    costs.add("training_teacher_prompt_tokens",sum(len(r["prompt_ids"]) for r in records)*len(terms))
    costs.add("training_reference_prompt_tokens",sum(len(r["prompt_ids"]) for r in records)*int(bool(ref_c)))
    return dict(update=index+1,u=u,loss=loss_total,gradient_norm=float(gradient),tokens=tokens),records


def run(config_path, *, resume=False):
    import time
    from .mpc_runtime import load_hidden,independent_copy,TeacherPool,restore_optimizer_exact,generate
    from .mpc_data import _inside,_rows,checkpoint_metadata
    from .mpc_allocation import validate_allocation
    from .mpc_ledger import CostLedger,snapshot_journals,rollback_journals,allocation_seconds
    from .mistral_runtime import load_tokenizers,assert_exact_tokenizer_pair
    from .optimizer import FP32AdamW
    from .training_common import (write_json,seed_everything,snapshot_rng,restore_rng,
                                  DistributedContext,save_model_only)
    cfg=json.loads(Path(config_path).read_text())
    validate_config(cfg)
    root=Path(cfg["root"]).resolve()
    if Path.cwd().resolve()!=root or str(root)!="/blue/du.j/jinjiaguo/CAFD":
        raise RuntimeError("runner must stay in authorized CAFD directory")
    if int(os.environ.get("WORLD_SIZE","1"))!=1 or not torch.cuda.is_available():
        raise RuntimeError("one visible B200 in an existing Slurm allocation required")
    allocation=validate_allocation(os.environ,torch.cuda.get_device_name(0),torch.cuda.device_count())
    allocation=validate_protocol_allocation(allocation,cfg)
    out=_inside(root,cfg["output_directory"])
    if (out/"complete.json").exists() or (out.exists() and not resume):
        raise RuntimeError("refusing overwrite; explicit --resume required for an existing run")
    if resume and json.loads((out/"config.json").read_text())!=cfg:
        raise ValueError("resume configuration changed")
    for path in [cfg["student_base"]]+cfg["route"]:
        checkpoint_metadata(root,path)
    fit=_rows(_inside(root,cfg["training_data"]))
    dev=_rows(_inside(root,cfg["development_data"]))
    if [r["id"] for r in fit]!=cfg["training_ids"] or [r["id"] for r in dev]!=cfg["development_ids"]:
        raise ValueError("frozen dataset ID order changed")
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"config.json",cfg)
    write_json(out/"allocation.json",allocation)
    costs=CostLedger(out/"physical_costs.jsonl")
    costs.begin_attempt(os.environ["SLURM_JOB_ID"],resume=resume)
    started=time.monotonic()
    device=torch.device("cuda:0")
    seed_everything(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    teacher_tok,tokenizer=load_tokenizers(root/".cache/huggingface/hub")
    assert_exact_tokenizer_pair(teacher_tok,tokenizer)
    student=load_hidden(root,cfg["student_base"],device,trainable=True)
    reference=independent_copy(student) if cfg["target"]=="cumulative_relative" else None
    pool=TeacherPool(root,device)
    optimizer=FP32AdamW(student.parameters(),lr=cfg["learning_rate"],betas=tuple(cfg["betas"]),eps=cfg["eps"],weight_decay=cfg["weight_decay"])
    context=DistributedContext(0,0,1,device)
    journals=[out/name for name in ("raw_rollouts.jsonl","training_curve.jsonl","development_outputs.jsonl","development_curve.jsonl")]
    def append(path,item):
        with path.open("a",encoding="utf-8") as f:
            f.write(json.dumps(item,sort_keys=True,allow_nan=False)+"\n")
            f.flush()
    start,best=0,None
    if resume:
        saved=torch.load(out/"resume.pt",map_location="cpu",weights_only=False)
        if saved["config"]!=cfg:
            raise ValueError("saved resume config changed")
        student.load_state_dict(saved["student"],strict=True)
        restore_optimizer_exact(optimizer,saved["optimizer"])
        start,best=saved["completed_updates"],saved["best"]
        rollback_journals(saved["journal_offsets"],[out],out/"discarded_attempts")
        restore_rng(saved["rng"])
        del saved
    def save_resume(step):
        temporary=out/"resume.tmp"
        torch.save(dict(config=cfg,completed_updates=step,best=best,student=student.state_dict(),
            optimizer=optimizer.state_dict(),rng=snapshot_rng(),journal_offsets=snapshot_journals(journals)),temporary)
        os.replace(temporary,out/"resume.pt")
    def evaluate(step,checkpoint):
        outputs=[]
        for i,row in enumerate(dev):
            record=generate(student,tokenizer,row,count=1,sample=False,
                max_new_tokens=cfg["max_new_tokens"],max_prompt_tokens=cfg["max_prompt_tokens"],seed=cfg["seed"]+2_000_000+i)[0]
            outputs.append(record)
            append(journals[2],dict(update=step,**record))
        costs.add("development_generated_tokens",sum(len(r["completion_ids"]) for r in outputs))
        costs.add("development_verifier_calls",len(outputs))
        result=dict(update=step,correct=sum(r["full_pass"] for r in outputs),total=len(outputs),checkpoint=str(checkpoint))
        append(journals[3],result)
        return result
    if not resume:
        best=evaluate(0,cfg["student_base"])
        save_resume(0)
    for index in range(start,cfg["total_updates"]):
        begin=time.monotonic()
        log,records=train_update(student,reference,pool,optimizer,tokenizer,
            [fit[j] for j in cfg["prompt_stream"][index]["indices"]],cfg,index,costs,device)
        log["elapsed_seconds"]=time.monotonic()-begin
        for r in records:
            append(journals[0],dict(update=index+1,**r))
        append(journals[1],log)
        step=index+1
        if step in cfg["milestones"]:
            checkpoint=out/f"step{step}"
            if checkpoint.exists():
                # A checkpoint written after the last committed resume belongs to
                # an interrupted attempt. Preserve it rather than overwrite.
                archived=out/"discarded_attempts"/(checkpoint.name+"-"+str(time.time_ns()))
                archived.parent.mkdir(exist_ok=True)
                checkpoint.rename(archived)
            save_model_only(student,tokenizer,checkpoint,context)
            result=evaluate(step,checkpoint)
            if result["correct"]>best["correct"]:
                best=result
            save_resume(step)
        write_json(out/"state.json",dict(status="training",update=step,u=cfg["u_by_update"][index]))
        costs["last_attempt_elapsed_seconds"]=time.monotonic()-started
        allocated=allocation_seconds(costs.job_ids)
        costs["gpu_hours"]=None if allocated is None else allocated/3600.
        costs["peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
        write_json(out/"costs.json",dict(costs))
    write_json(out/"selected.json",dict(status="frozen",selection_rule=cfg["selection_rule"],**best))
    write_json(out/"complete.json",dict(status="completed",updates=200,frozen_test_evaluated=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--execute-training",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    validate_config(cfg)
    if args.execute_training:
        run(args.config,resume=args.resume)
    else:
        print(json.dumps(dict(status="CONFIG_VALIDATED_NOT_SUBMITTED",target=cfg["target"],updates=200)))
