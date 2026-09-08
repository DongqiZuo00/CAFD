"""CAFD-MPC integration runner, independent of all previous CAFD variants.

Default smoke profile is an engineering test, not a formal result. Formal uses
200 rounds, 4x8 samples, 2048-token cap, K=10/H=2, and never reads test labels.
"""
import argparse
import copy
import gc
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .mpc_controller import MPCConfig
from .kd_retention_replay import ReplayController as MPCController, U_BY_BLOCK
from .mpc_data import prepare, load_mpc_rows
from .kd_retention_objective import (teacher_coefficients, route_reward_groups, exact_linear_forward_kl,
                            exact_linear_token_log_probs, clipped_rl_loss)
from .kd_retention_runtime import (load_hidden, independent_copy, refresh_behavior, TeacherPool,
                         generate, sequence_inputs, frozen_sources, target_chunks, restore_optimizer_exact)
from .mistral_runtime import load_tokenizers, assert_exact_tokenizer_pair
from .prompting import assert_matches_mistral_chat_template
from .optimizer import FP32AdamW
from .mpc_allocation import validate_allocation
from .mpc_ledger import CostLedger, snapshot_journals, rollback_journals, allocation_seconds
from .data import prompt_stream, rollout_seed
from .training_common import (write_json, snapshot_rng, restore_rng, save_model_only,
                              DistributedContext, seed_everything)

ROOT = Path("/blue/du.j/jinjiaguo/CAFD")
S0 = ROOT/"runs/cafd/experiments/mistral_cafd_disjoint_v7/student_base/S0"
ROUTE = ROOT/"runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher/route.json"

def configuration(profile):
    cfg = dict(protocol="CAFD-MPC-2026-09-06", profile=profile, seed=2027, total_rounds=200,
        prompts_per_round=4, rollouts_per_prompt=8, max_new_tokens=2048, max_prompt_tokens=4096,
        block_rounds=10, horizon=2, learning_rate=1e-6, token_chunk=64, grad_clip=1.,
        control_per_family=4, control_rollouts=2, milestones=[0,10,40,80,120,160,200],
        reference="permanent_initial_student", support="current_student_only",
        target="S0+(1-alpha)*Tm+alpha*Tnext-T0", temperature=1., top_p=1., top_k=0,
        output_stop="external_closing_fence_or_generated_EOS_no_synthetic_EOS",
        reward="hierarchical_0_.05_.10_.10+.90*pass_rate_1",
        optimizer="FP32-master-AdamW", lr_clock="rollout_round", lr_schedule="constant",
        lambda_rl=1., advantage_epsilon=1e-6, ratio_clip=.2,
        frozen_test_evaluated=False, maximum_gpus=2, reserved_memory_gib=192)
    if profile == "smoke":
        cfg.update(total_rounds=8, prompts_per_round=2, rollouts_per_prompt=2,
                   max_new_tokens=64, block_rounds=1, control_per_family=1,
                   milestones=[], not_a_formal_result=True)
    cfg.update(method="KD coverage modification + old schedule replay", kd_retention=True,
               u_by_block=U_BY_BLOCK, maximum_gpus=1, baseline_run="mistral_cafd_mpc_v1_formal")
    if profile == "smoke":
        cfg.update(total_rounds=2)
    return cfg

def append_jsonl(path, item):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(item, sort_keys=True, allow_nan=False)+"\n")
        handle.flush()

def add(costs, name, value):
    if isinstance(costs, CostLedger):
        costs.add(name, value)
    else:
        costs[name] = costs.get(name, 0) + value

def count_generation(costs, records, purpose):
    add(costs, purpose+"_generated_tokens", sum(len(x["completion_ids"]) for x in records))
    add(costs, purpose+"_prompt_tokens", sum(len(x["prompt_ids"]) for x in records))
    add(costs, "verifier_calls", len(records))
    add(costs, "generation_length_caps", sum(x["length_cap"] for x in records))

def count_sources(costs, records, coefficients, purpose):
    tokens = sum(len(r["completion_ids"]) for r in records)
    prompts = sum(len(r["prompt_ids"]) for r in records)
    add(costs, purpose+"_teacher_scored_tokens", tokens*len(coefficients))
    add(costs, purpose+"_reference_scored_tokens", tokens)
    add(costs, purpose+"_teacher_prompt_tokens", prompts*len(coefficients))
    add(costs, purpose+"_reference_prompt_tokens", prompts)

def build_probe_cache(out, control, reference, pool, route, tokenizer, cfg, costs, device):
    manifest_path = out/"fixed_probes/manifest.json"
    if manifest_path.exists():
        result = json.loads(manifest_path.read_text())
        if result["control_ids"] != [r["id"] for r in control] or result["route"] != route:
            raise RuntimeError("fixed probe identity changed")
        for item in result["probes"]:
            for chunk in item["chunks"]:
                if not (out/chunk["path"]).is_file() or (out/chunk["path"]).stat().st_size != chunk["bytes"]:
                    raise RuntimeError("fixed qM cache missing; cannot silently replace probes")
        return result
    coefficients = teacher_coefficients(len(route)-1, route)
    teachers = pool.select(coefficients)
    probes = []
    for index,row in enumerate(control):
        record = generate(reference, tokenizer, row, count=1, max_new_tokens=cfg["max_new_tokens"],
            seed=cfg["seed"]+100_000+index, max_prompt_tokens=cfg["max_prompt_tokens"])[0]
        count_generation(costs, [record], "probe_setup")
        sources = frozen_sources(record, tokenizer, reference, teachers, coefficients, device)
        if not sources[0][1].shape[0]:
            raise RuntimeError("empty fixed prefix set")
        chunks = []
        for start,end,q in target_chunks(sources, cfg["token_chunk"]):
            relative = f"fixed_probes/probe{index:02d}/q-{start:05d}.pt"
            path = out/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            torch.save(q.cpu(), temporary)
            os.replace(temporary, path)
            chunks.append(dict(start=start,end=end,path=relative,bytes=path.stat().st_size))
        count_sources(costs,[record],coefficients,"probe_setup")
        probes.append(dict(record=record,chunks=chunks))
        del sources
    result = dict(version=1, route=route, reference=str(S0), dtype="float32", full_vocabulary=True,
                  control_ids=[r["id"] for r in control], probes=probes,
                  bytes=sum(c["bytes"] for p in probes for c in p["chunks"]))
    write_json(manifest_path, result)
    return result

@torch.no_grad()
def observe_once(student, tokenizer, control, probe_cache, families, out, cfg, boundary, costs, device):
    previous_mode = student.training
    student.eval()
    try:
        fresh = []
        for index,row in enumerate(control):
            records = generate(student,tokenizer,row,count=cfg["control_rollouts"],
                max_new_tokens=cfg["max_new_tokens"], seed=cfg["seed"]+1_000_000+boundary*10_000+index,
                max_prompt_tokens=cfg["max_prompt_tokens"])
            count_generation(costs,records,"control")
            fresh.extend(records)
        sums,counts = Counter(),Counter()
        for probe in probe_cache["probes"]:
            record = probe["record"]
            ids,attention,mask = sequence_inputs(record,tokenizer,device)
            hidden = student(ids,attention)[:, :-1][mask]
            head = student.lm_head
            for chunk in probe["chunks"]:
                q = torch.load(out/chunk["path"],map_location=device,weights_only=True)
                h = hidden[chunk["start"]:chunk["end"]]
                if q.dtype != torch.float32 or q.shape != (len(h),head.weight.shape[0]):
                    raise RuntimeError("fixed qM cache shape/dtype mismatch")
                if not torch.isfinite(q).all() or (q < 0).any() or not torch.allclose(q.sum(-1),torch.ones(len(q),device=device),atol=2e-5,rtol=0):
                    raise RuntimeError("fixed qM cache is not a full-vocabulary probability distribution")
                with torch.autocast(device_type=device.type,enabled=False):
                    logp = F.linear(h.float(),head.weight.float(),
                                   None if head.bias is None else head.bias.float()).log_softmax(-1)
                    value = (torch.special.xlogy(q,q)-q*logp).sum()
                if not torch.isfinite(value):
                    raise RuntimeError("non-finite endpoint control KL")
                sums[record["family"]] += float(value)
                counts[record["family"]] += len(q)
                del q,logp
            add(costs,"control_student_scored_tokens",len(hidden))
            add(costs,"control_student_prompt_tokens",len(record["prompt_ids"]))
            del hidden
        states = []
        for family in families:
            rows = [r for r in fresh if r["family"]==family]
            if not rows or not counts[family]:
                raise RuntimeError("incomplete control family")
            error = max(0.,sums[family]/counts[family])
            states.append([sum(r["reward"] for r in rows)/len(rows),
                           sum(r["full_pass"] for r in rows)/len(rows),error/(1.+error)])
        for record in fresh:
            append_jsonl(out/"control_outputs.jsonl",dict(boundary=boundary,**record))
        append_jsonl(out/"observations.jsonl",dict(boundary=boundary,state=states,status="complete"))
        return np.asarray(states)
    finally:
        student.train(previous_mode)

def observe(*args, **kwargs):
    # Infrastructure misses are not semantic failures and must never become KL=0.
    for attempt in range(2):
        try:
            return observe_once(*args, **kwargs)
        except (OSError,TimeoutError,ConnectionError) as error:
            out = args[5]
            append_jsonl(out/"infrastructure_errors.jsonl",dict(attempt=attempt,boundary=args[7],error=repr(error)))
    append_jsonl(args[5]/"observations.jsonl",dict(boundary=args[7],status="missing",attempts=2,state=None))
    return None

def train_round(student,behavior,reference,pool,route,optimizer,tokenizer,rows,cfg,round_index,u,costs,device):
    rollout_started=time.monotonic()
    refresh_behavior(behavior,student)
    groups = []
    for index,row in enumerate(rows):
        records = generate(behavior,tokenizer,row,count=cfg["rollouts_per_prompt"],
            max_new_tokens=cfg["max_new_tokens"],seed=rollout_seed(round_index,index,0,cfg["seed"]),
            max_prompt_tokens=cfg["max_prompt_tokens"])
        groups.append(records)
        count_generation(costs,records,"training")
    add(costs,"rollout_seconds",time.monotonic()-rollout_started)
    reward = torch.tensor([[r["reward"] for r in group] for group in groups],device=device,dtype=torch.float64)
    verified = torch.tensor([[r["full_pass"] for r in group] for group in groups],device=device)
    routing = route_reward_groups(reward,full_pass=verified, kd_retention=cfg.get("kd_retention",False))
    z = sum(len(r["completion_ids"]) for group in groups for r in group)
    log = dict(round=round_index+1,u=u,normalization_tokens=z,distill_groups=int(routing.distill.sum()),
               rl_groups=int(routing.rl.sum()),skip_groups=int(routing.skip.sum()),optimizer_step=False,
               kd_loss=0.,rl_loss=0.,gradient_norm=0.,max_behavior_logprob_delta=0.)
    lengths=[sum(len(r["completion_ids"]) for r in g) for g in groups]
    kd=[bool(x) for x in routing.distill]; rl=[bool(x) for x in routing.rl]
    old_routing=route_reward_groups(reward,full_pass=verified)
    log.update(block=round_index//cfg["block_rounds"]+1,
        kd_only_groups=sum(d and not r for d,r in zip(kd,rl)),
        rl_only_groups=sum(r and not d for d,r in zip(kd,rl)),
        kd_rl_groups=sum(d and r for d,r in zip(kd,rl)),
        all_fail_variable_groups=sum((not any(x["full_pass"] for x in g)) and max(x["reward"] for x in g)>min(x["reward"] for x in g) for g in groups),
        kd_tokens=sum(n for n,d in zip(lengths,kd) if d),
        rl_tokens=sum(n for n,r in zip(lengths,rl) if r),
        overlap_tokens=sum(n for n,d,r in zip(lengths,kd,rl) if d and r))
    log["group_routing"]=[dict(round=round_index+1,block=log["block"],row_id=g[0]["row_id"],
        family=g[0]["family"],u=u,rewards=[x["reward"] for x in g],
        full_pass=[x["full_pass"] for x in g],
        reward_range=max(x["reward"] for x in g)-min(x["reward"] for x in g),
        use_kd=kd[j],use_rl=rl[j],valid_tokens=lengths[j],
        completion_tokens=[len(x["completion_ids"]) for x in g],
        old_route="KD" if bool(old_routing.distill[j]) else ("RL" if bool(old_routing.rl[j]) else "skip"))
        for j,g in enumerate(groups)]
    if not z or not routing.should_step:
        return log,groups
    coefficients = teacher_coefficients(u,route)
    scoring_started=time.monotonic()
    teachers = pool.select(coefficients) if bool(routing.distill.any()) else {}
    add(costs,"teacher_loading_seconds",time.monotonic()-scoring_started)
    optimizer.zero_grad(set_to_none=True)
    student.train()
    contributing = 0
    for j,group in enumerate(groups):
        if bool(routing.skip[j]):
            continue
        for l,record in enumerate(group):
            ids,attention,mask = sequence_inputs(record,tokenizer,device)
            if not bool(mask.any()):
                continue
            contributing += 1
            # Frozen models are scored only for the selected distillation groups.
            scoring_started=time.monotonic()
            sources = frozen_sources(record,tokenizer,reference,teachers,coefficients,device) if bool(routing.distill[j]) else None
            add(costs,"frozen_scoring_seconds",time.monotonic()-scoring_started)
            optimization_started=time.monotonic()
            hidden = student(ids,attention)[:, :-1][mask]
            loss = None
            head = student.lm_head
            if sources is not None:
                loss = exact_linear_forward_kl(hidden,head.weight,sources,normalization_tokens=z,
                                               token_chunk=cfg["token_chunk"],student_bias=head.bias)
                log["kd_loss"] += float(loss.detach())
                count_sources(costs,[record],coefficients,"training")
            if bool(routing.rl[j]):
                labels = ids[:,1:][mask]
                current = exact_linear_token_log_probs(hidden,head.weight,labels,
                    token_chunk=cfg["token_chunk"],student_bias=head.bias)
                old = torch.tensor(record["old_log_probs"],device=device,dtype=torch.float32)
                if current.shape != old.shape:
                    raise RuntimeError("behavior log-probability / completion mask mismatch")
                log["max_behavior_logprob_delta"] = max(log["max_behavior_logprob_delta"],float((current.detach()-old).abs().max()))
                rl_loss = clipped_rl_loss(current,old,routing.advantages[j,l],torch.ones_like(current,dtype=torch.bool),
                    normalization_tokens=z,clip=cfg["ratio_clip"],lambda_rl=cfg["lambda_rl"])
                log["rl_loss"] += float(rl_loss.detach())
                loss = rl_loss if loss is None else loss + rl_loss
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            add(costs,"optimization_forward_backward_seconds",time.monotonic()-optimization_started)
            del loss,hidden,sources
    if contributing:
        step_started=time.monotonic()
        norm = torch.nn.utils.clip_grad_norm_(student.parameters(),cfg["grad_clip"],error_if_nonfinite=True)
        optimizer.step()
        optimizer.assert_fp32_states()
        add(costs,"optimizer_step_seconds",time.monotonic()-step_started)
        log.update(optimizer_step=True,gradient_norm=float(norm))
    optimizer.zero_grad(set_to_none=True)
    if any(p.grad is not None for p in reference.parameters()) or any(p.grad is not None for t in teachers.values() for p in t.parameters()):
        raise RuntimeError("frozen target received a gradient")
    return log,groups

@torch.no_grad()
def development(student,tokenizer,rows,cfg,step,costs):
    development_started=time.monotonic()
    records = []
    for index,row in enumerate(rows):
        sampled = generate(student,tokenizer,row,count=1,max_new_tokens=cfg["max_new_tokens"],
                           seed=cfg["seed"]+2_000_000+index,sample=False,max_prompt_tokens=cfg["max_prompt_tokens"])
        count_generation(costs,sampled,"development")
        records.extend(sampled)
    correct = sum(r["full_pass"] for r in records)
    add(costs,"development_seconds",time.monotonic()-development_started)
    allocated=allocation_seconds(costs.job_ids) if isinstance(costs,CostLedger) else None
    return dict(round=step,correct=correct,total=len(records),accuracy=correct/len(records),
        mean_reward=sum(r["reward"] for r in records)/len(records),
        cumulative_allocated_gpu_hours=None if allocated is None else allocated/3600,
        evaluated_at=time.time()),records

def journal_paths(out,artifacts):
    return [out/name for name in ("raw_rollouts.jsonl","controller_decisions.jsonl","control_outputs.jsonl",
             "observations.jsonl","transitions.jsonl","development_outputs.jsonl")] + [artifacts/name for name in ("training_curve.jsonl","development_curve.jsonl")]

def save_resume(out,student,optimizer,controller,costs,cfg,best,actual_updates,artifacts):
    temporary = out/"resume.tmp"
    torch.save(dict(version=1,config=cfg,student=student.state_dict(),optimizer=optimizer.state_dict(),
        controller=controller.state_dict(),costs=dict(costs),best=best,actual_updates=actual_updates,
        journal_offsets=snapshot_journals(journal_paths(out,artifacts)), rng=snapshot_rng()),temporary)
    os.replace(temporary,out/"resume.pt")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--profile",choices=["smoke","formal"],default="smoke")
    parser.add_argument("--run-id",default=None)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--stop-after-round",type=int,default=None)
    args=parser.parse_args()
    args.run_id=args.run_id or ("mistral_cafd_kd_retention_replay_s2027"+("_smoke" if args.profile=="smoke" else ""))
    if args.profile=="formal" and args.run_id.endswith("_smoke"):
        raise ValueError("formal and smoke run identities must not be mixed")
    if Path.cwd().resolve()!=ROOT:
        raise RuntimeError("all work must remain in CAFD")
    if not re.fullmatch(r"mistral_cafd_kd_retention_[a-zA-Z0-9_]+",args.run_id):
        raise ValueError("new MPC run ID required")
    manifest=prepare(ROOT)
    if args.prepare_only:
        print(json.dumps(dict(status="preflight_passed",split="data/cafd/mpc_v1",test_labels_used=False)),flush=True)
        return
    if int(os.environ.get("WORLD_SIZE","1"))!=1:
        raise RuntimeError("this runner is explicitly single-GPU; no silent multi-device normalization")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU integration requires Slurm CUDA allocation")
    allocation=validate_allocation(os.environ,torch.cuda.get_device_name(0),torch.cuda.device_count())
    cfg=configuration(args.profile)
    out=ROOT/"runs/cafd/experiments"/args.run_id
    artifacts=ROOT/"artifacts/cafd/experiments"/args.run_id
    state_path=ROOT/"state/cafd/experiments"/args.run_id/"state.json"
    if (out/"complete.json").exists():
        print((out/"complete.json").read_text(),flush=True)
        return
    if (out/"config.json").exists() and not args.resume:
        raise RuntimeError("run already initialized; use explicit --resume or a new run ID")
    out.mkdir(parents=True,exist_ok=True)
    artifacts.mkdir(parents=True,exist_ok=True)
    if args.resume and json.loads((out/"config.json").read_text())!=cfg:
        raise RuntimeError("resume config differs")
    write_json(out/"config.json",cfg)
    write_json(out/"data_manifest.json",manifest)
    write_json(out/"allocation.json",allocation)
    started=time.monotonic()
    device=torch.device("cuda:0")
    torch.cuda.set_device(device)
    seed_everything(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    costs=CostLedger(artifacts/"physical_costs.jsonl")
    costs.begin_attempt(os.environ["SLURM_JOB_ID"],resume=args.resume)
    write_json(state_path,dict(status="loading",job_id=os.environ.get("SLURM_JOB_ID"),profile=args.profile))
    teacher_tok,tokenizer=load_tokenizers(ROOT/".cache/huggingface/hub")
    assert_exact_tokenizer_pair(teacher_tok,tokenizer)
    assert_matches_mistral_chat_template(tokenizer)
    route=[str(Path(r["checkpoint"]).resolve()) for r in json.loads(ROUTE.read_text())["checkpoints"]]
    optimization=load_mpc_rows(ROOT,"optimization")
    control=load_mpc_rows(ROOT,"control")
    dev=load_mpc_rows(ROOT,"development")
    families=sorted({r["problem_family"] for r in control})
    control=[r for family in families for r in [x for x in control if x["problem_family"]==family][:cfg["control_per_family"]]]
    mixture=Counter(r["problem_family"] for r in optimization)
    weights=tuple(mixture[f]/len(optimization) for f in families)
    student=load_hidden(ROOT,S0,device,trainable=True)
    assert_exact_tokenizer_pair(teacher_tok,tokenizer,student_model=student.causal_lm)
    # Distinct permanent initial reference and distinct per-round behavior policy.
    reference=independent_copy(student)
    behavior=independent_copy(student)
    optimizer=FP32AdamW(student.parameters(),lr=cfg["learning_rate"],weight_decay=0.)
    pool=TeacherPool(ROOT,device)
    best=None
    actual_updates=0
    prior_elapsed=0.
    prior_loads=0
    expected_controller_config=MPCConfig(route_intervals=len(route)-1,total_rounds=cfg["total_rounds"],
        family_weights=weights,seed=cfg["seed"],block_rounds=cfg["block_rounds"],horizon=cfg["horizon"])
    if args.resume:
        saved=torch.load(out/"resume.pt",map_location="cpu",weights_only=False)
        if saved["config"]!=cfg:
            raise RuntimeError("resume payload config differs")
        student.load_state_dict(saved["student"],strict=True)
        restore_optimizer_exact(optimizer,saved["optimizer"])
        write_json(out/"resume_validation.json",dict(
            loaded_round=saved["controller"]["completed_rounds"],
            actual_updates=saved["actual_updates"],optimizer_fp32_validated=True,
            permanent_reference=str(S0),reference_initialized_before_resume=True,
            distinct_roles=all(a.data_ptr()!=b.data_ptr() and a.data_ptr()!=c.data_ptr() and b.data_ptr()!=c.data_ptr()
                for a,b,c in zip(student.parameters(),reference.parameters(),behavior.parameters()))))

        controller=MPCController.from_state_dict(saved["controller"])
        if controller.config!=expected_controller_config or controller.pending is not None:
            raise RuntimeError("resume controller/config is inconsistent; only committed block boundaries supported")
        best,actual_updates=saved["best"],saved["actual_updates"]
        prior_elapsed=saved["costs"].get("elapsed_seconds",0.)
        prior_loads=saved["costs"].get("teacher_model_loads",0)
        rollback_journals(saved["journal_offsets"],[out,artifacts],out/"discarded_attempts")
        # Preserve uncommitted checkpoints as failed-attempt evidence, never silently reuse them.
        for path in out.iterdir():
            if re.fullmatch(r"round[0-9]+",path.name) and int(path.name[5:])>controller.completed_rounds:
                destination=out/"discarded_attempts"/(path.name+"-"+str(time.time_ns()))
                destination.parent.mkdir(exist_ok=True)
                path.rename(destination)
        restore_rng(saved["rng"])
        del saved
        gc.collect()
    probe_started=time.monotonic()
    probes=build_probe_cache(out,control,reference,pool,route,tokenizer,cfg,costs,device)
    costs["fixed_target_cache_bytes"]=probes["bytes"]
    add(costs,"probe_initialization_seconds",time.monotonic()-probe_started)
    if not args.resume:
        control_started=time.monotonic()
        initial=observe(student,tokenizer,control,probes,families,out,cfg,0,costs,device)
        add(costs,"control_seconds",time.monotonic()-control_started)
        controller=MPCController(expected_controller_config,initial)
    context=DistributedContext(0,0,1,device)
    stream=prompt_stream(optimization,cfg["total_rounds"],cfg["prompts_per_round"],cfg["seed"])
    if args.profile=="formal" and best is None:
        result,records=development(student,tokenizer,dev,cfg,0,costs)
        best=dict(**result,checkpoint=str(S0.resolve()))
        append_jsonl(artifacts/"development_curve.jsonl",result)
        for record in records:
            append_jsonl(out/"development_outputs.jsonl",dict(round=0,**record))
        write_json(out/"selection_in_progress.json",best)
    if not args.resume:
        save_resume(out,student,optimizer,controller,costs,cfg,best,actual_updates,artifacts)
    write_json(out/"prompt_stream.json",[[optimization[i]["id"] for i in batch] for batch in stream])
    add(costs,"initialization_inclusive_seconds",time.monotonic()-started)
    while not controller.done:
        t=time.perf_counter()
        decision=controller.choose_action()
        add(costs,"controller_cpu_seconds",time.perf_counter()-t)
        append_jsonl(out/"controller_decisions.jsonl",decision)
        before=controller.completed_rounds
        for local in range(decision["rounds"]):
            round_index=before+local
            t=time.monotonic()
            log,groups=train_round(student,behavior,reference,pool,route,optimizer,tokenizer,
                [optimization[i] for i in stream[round_index]],cfg,round_index,decision["u"],costs,device)
            actual_updates+=int(log["optimizer_step"])
            log["elapsed_seconds"]=time.monotonic()-t
            for group in groups:
                for record in group:
                    append_jsonl(out/"raw_rollouts.jsonl",dict(round=round_index+1,**record))
            append_jsonl(artifacts/"training_curve.jsonl",log)
            write_json(state_path,dict(status="training",round=round_index+1,actual_updates=actual_updates,
                u=decision["u"],job_id=os.environ.get("SLURM_JOB_ID"),profile=args.profile))
            if round_index+1 in cfg["milestones"]:
                result,records=development(student,tokenizer,dev,cfg,round_index+1,costs)
                append_jsonl(artifacts/"development_curve.jsonl",result)
                for record in records:
                    append_jsonl(out/"development_outputs.jsonl",dict(round=round_index+1,**record))
                if best is None or result["correct"]>best["correct"]:
                    checkpoint=out/f"round{round_index+1}"
                    save_model_only(student,tokenizer,checkpoint,context)
                    best=dict(**result,checkpoint=str(checkpoint))
                    write_json(out/"selection_in_progress.json",best)
        boundary=controller.completed_full_blocks+1
        control_started=time.monotonic()
        next_state=observe(student,tokenizer,control,probes,families,out,cfg,boundary,costs,device)
        add(costs,"control_seconds",time.monotonic()-control_started)
        t=time.perf_counter()
        transition=controller.observe(next_state)
        add(costs,"controller_cpu_seconds",time.perf_counter()-t)
        append_jsonl(out/"transitions.jsonl",transition)
        write_json(out/"controller_state.json",controller.state_dict())
        costs["elapsed_seconds"]=prior_elapsed+time.monotonic()-started
        costs["gpu_hours"]=costs["elapsed_seconds"]/3600.
        costs["peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
        costs["teacher_model_loads"]=prior_loads+pool.loads
        costs["actual_optimizer_updates"]=actual_updates
        costs["rollout_rounds"]=controller.completed_rounds
        save_started=time.monotonic()
        save_resume(out,student,optimizer,controller,costs,cfg,best,actual_updates,artifacts)
        add(costs,"save_resume_seconds",time.monotonic()-save_started)
        costs["elapsed_seconds"]=prior_elapsed+time.monotonic()-started
        allocated_seconds=allocation_seconds(costs.job_ids)
        costs["gpu_hours"]=(allocated_seconds if allocated_seconds is not None else costs["elapsed_seconds"])/3600.
        costs["gpu_time_basis"]="Slurm_allocated_elapsed" if allocated_seconds is not None else "worker_elapsed_lower_bound"
        write_json(artifacts/"costs.json",dict(costs))
        if args.stop_after_round is not None and controller.completed_rounds>=args.stop_after_round and not controller.done:
            print(json.dumps(dict(status="VALIDATION_PAUSED",rounds=controller.completed_rounds,actual_updates=actual_updates)),flush=True)
            return
    if actual_updates != cfg["total_rounds"]:
        raise RuntimeError("round budget completed without required optimizer updates")
    if best is not None:
        write_json(out/"selected.json",dict(status="frozen",selection_rule="development_full_pass_earliest_tie",**best))
    result=dict(status="SMOKE_PASSED" if args.profile=="smoke" else "STUDENT_TRAINING_COMPLETE",
        formal_result=args.profile=="formal",rounds=controller.completed_rounds,actual_updates=actual_updates,
        predictive_decisions=controller.predictive_decisions,multiaction_mpc_decisions=controller.mpc_multiaction_decisions,
        final_u=controller.u,selected=best,frozen_test_evaluated=False,costs=dict(costs),
        job_id=os.environ.get("SLURM_JOB_ID"),node=os.environ.get("SLURMD_NODENAME"),
        gpu=torch.cuda.get_device_name(device),data_limitations=manifest.get("limitations",[]))
    write_json(out/"complete.json",result)
    write_json(artifacts/"result.json",result)
    write_json(state_path,result)
    print(json.dumps(result),flush=True)

if __name__=="__main__":
    main()
