"""Aggregate only the predeclared T1 selection and T2 fit-prefix diagnostics.

This program never reads frozen-test files. Missing planned episodes are
explicitly materialized and cannot silently change any denominator. Bootstrap
resamples question IDs, keeping both paired stochastic replicates together.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def write_json(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        path.write_text("",encoding="utf-8")
        return
    with path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def planned_policies(M=5):
    return [(f"{letter}{stage}_random","random",2,stage,letter)
            for stage in range(M+1) for letter in ("q","p")] + [
            ("q0_greedy","greedy",1,0,"q"),(f"q{M}_greedy","greedy",1,M,"q"),
            (f"p{M}_greedy","greedy",1,M,"p")]


def collect_policy(run_dir, policy, selection):
    policy_id,mode,count,stage,kind=policy
    raw=read_jsonl(run_dir/"policies"/policy_id/"rollouts.jsonl")
    ids={str(row["id"]) for row in selection}
    found={}
    for record in raw:
        key=(str(record["row_id"]),int(record.get("sample_index",record.get("replicate",-1))))
        if key[0] not in ids or not 0<=key[1]<count:
            raise ValueError(f"unexpected episode {policy_id}: {key}")
        if key in found:
            raise ValueError(f"duplicate episode {policy_id}: {key}")
        if record.get("policy_id") != policy_id:
            raise ValueError(f"policy identity mismatch in {policy_id}")
        if bool(record.get("sample",mode=="random")) != (mode=="random"):
            raise ValueError(f"decoding mode mismatch in {policy_id}")
        status=record.get("status","error" if record.get("error") else "ok")
        if status not in ("ok","error"):
            raise ValueError(f"invalid persisted episode status: {status}")
        reward=record.get("reward")
        if not isinstance(reward,(int,float)) or not math.isfinite(reward) or not 0<=reward<=1:
            raise ValueError(f"invalid reward {policy_id}: {key}")
        if record.get("full_pass") and (status!="ok" or reward!=1.):
            raise ValueError(f"inconsistent success {policy_id}: {key}")
        if int(record["generated_tokens"]) != len(record["completion_ids"]):
            raise ValueError(f"token count mismatch {policy_id}: {key}")
        found[key]=dict(record,status=status,mode=mode,sample_index=key[1])
    records=[]
    for task in selection:
        for replicate in range(count):
            key=(str(task["id"]),replicate)
            record=found.get(key)
            if record is None:
                record=dict(policy_id=policy_id,mode=mode,row_id=key[0],sample_index=replicate,
                    family=task["problem_family"],seed=None,status="missing",error="planned_episode_not_yet_recorded",
                    completion_text=None,completion_ids=[],generated_tokens=None,reward=None,full_pass=False,
                    tier=None,truncated=None,format_error=None,parse_error=None,verifier=None)
            elif record["family"]!=task["problem_family"]:
                raise ValueError(f"family mismatch {policy_id}: {key}")
            records.append(dict(record,stage=stage,policy_kind=kind))
    return records


def profile(records, policy, costs):
    policy_id,mode,count,stage,kind=policy
    result=[]
    families=["all"]+sorted({r["family"] for r in records})
    for family in families:
        rows=records if family=="all" else [r for r in records if r["family"]==family]
        present=[r for r in rows if r["status"]!="missing"]
        N=len(rows)
        missing=N-len(present)
        correct=sum(bool(r["full_pass"]) for r in present)
        error=sum(r["status"]=="error" for r in present)
        rewards=sum(float(r["reward"]) if r["status"]=="ok" else 0. for r in present)
        tokens=sum(int(r["generated_tokens"]) for r in present)
        fmt=sum(bool(r.get("format_error")) for r in present)
        parse=sum(bool(r.get("parse_error")) for r in present)
        trunc=sum(bool(r.get("truncated")) for r in present)
        semantic=[r for r in present if r["status"]=="ok" and not r["full_pass"]
                  and not r.get("format_error") and not r.get("parse_error")]
        complete=missing==0
        result.append(dict(policy_id=policy_id,mode=mode,stage=stage,policy_kind=kind,family=family,
            status=("contaminated" if error else "complete") if complete else "incomplete",expected_questions=N//count,replicates=count,
            correct=correct,N=N,recorded=len(present),missing=missing,infrastructure_errors=error,
            full_pass_rate=correct/N if complete else None,
            full_pass_rate_known_lower_bound=correct/N,full_pass_rate_unknown_upper_bound=(correct+missing)/N,
            question_mean_success_sum=correct/count if complete else None,
            mean_partial_reward=rewards/N if complete else None,
            known_reward_lower_bound=rewards/N,
            format_errors=fmt,format_error_rate=fmt/N if complete else None,
            parse_errors=parse,parse_error_rate=parse/N if complete else None,
            valid_parse_semantic_failures=len(semantic),
            valid_parse_semantic_failures_truncated=sum(bool(r.get("truncated")) for r in semantic),
            valid_parse_semantic_failures_not_truncated=sum(not r.get("truncated") for r in semantic),
            truncations=trunc,truncation_rate=trunc/N if complete else None,
            generated_tokens=tokens,mean_generated_tokens=tokens/N if complete else None,
            mean_generated_tokens_recorded=tokens/len(present) if present else None,
            generation_seconds=costs.get("generation_seconds") if family=="all" else None,
            model_scored_tokens=costs.get("model_scored_tokens") if family=="all" else None,
            gpu_hours=costs["gpu_wall_seconds"]/3600 if family=="all" and "gpu_wall_seconds" in costs else None))
    return result


def paired_bootstrap(q_records,p_records,*,replicates=2,draws=10000,seed=2027):
    """Each draw samples paired question means, never independent completions."""
    qids=sorted({r["row_id"] for r in q_records})
    if qids!=sorted({r["row_id"] for r in p_records}):
        raise ValueError("paired question IDs differ")
    if any(r["status"]=="missing" for r in q_records+p_records):
        return dict(status="incomplete",n_questions=len(qids),bootstrap_draws=draws,bootstrap_seed=seed,
            infrastructure_errors_q=sum(r["status"]=="error" for r in q_records),
            infrastructure_errors_p=sum(r["status"]=="error" for r in p_records),
            delta_success_rate=None,success_ci_low=None,success_ci_high=None,
            delta_partial_reward=None,reward_ci_low=None,reward_ci_high=None)
    q_seed={(r["row_id"],r["sample_index"]):r.get("seed") for r in q_records}
    p_seed={(r["row_id"],r["sample_index"]):r.get("seed") for r in p_records}
    if q_seed!=p_seed:
        raise ValueError("paired policies did not use the same per-episode seed table")
    def vectors(records):
        success,reward=[],[]
        for row_id in qids:
            rows=[r for r in records if r["row_id"]==row_id]
            if len(rows)!=replicates or {r["sample_index"] for r in rows}!=set(range(replicates)):
                raise ValueError("incomplete or duplicate per-question replicates")
            success.append(sum(bool(r["full_pass"]) for r in rows)/replicates)
            reward.append(sum(float(r["reward"]) if r["status"]=="ok" else 0. for r in rows)/replicates)
        return np.array(success),np.array(reward)
    qs,qr=vectors(q_records)
    ps,pr=vectors(p_records)
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,len(qids),size=(draws,len(qids)))
    ds,dr=qs-ps,qr-pr
    bs,br=ds[indices].mean(1),dr[indices].mean(1)
    q_errors=sum(r["status"]=="error" for r in q_records)
    p_errors=sum(r["status"]=="error" for r in p_records)
    return dict(status="contaminated" if q_errors+p_errors else "complete",n_questions=len(qids),bootstrap_draws=draws,bootstrap_seed=seed,
        infrastructure_errors_q=q_errors,infrastructure_errors_p=p_errors,
        delta_success_rate=float(ds.mean()),success_ci_low=float(np.quantile(bs,.025)),
        success_ci_high=float(np.quantile(bs,.975)),delta_partial_reward=float(dr.mean()),
        reward_ci_low=float(np.quantile(br,.025)),reward_ci_high=float(np.quantile(br,.975)))


def _sum_costs(entries):
    fields=("generation_seconds","verifier_seconds","verifier_calls","model_load_seconds",
        "generated_tokens","model_forward_calls","transformer_token_positions","useful_prefill_tokens",
        "output_head_positions","active_output_head_positions","model_scored_tokens",
        "gpu_wall_seconds","total_wall_seconds")
    sums={key:sum(c.get(key,0.) for c in entries) for key in fields}
    sums["peak_allocated_bytes"]=max([0]+[c.get("peak_allocated_bytes",0) for c in entries])
    sums["peak_reserved_bytes"]=max([0]+[c.get("peak_reserved_bytes",0) for c in entries])
    return sums


def _fmt(value, digits=4):
    return "未齐" if value is None else f"{float(value):.{digits}f}"


def execution_accounting(run_dir,extra_job_ids=()):
    run_dir=run_dir.resolve()
    paths=list(run_dir.glob("worker*_state.json"))
    paths+=list((run_dir/"claim_archive").glob("**/worker_provenance/worker*_state.json"))
    # The copied destinations are also indexed by recovery provenance. Never
    # trust a recovery entry to send this reader outside the diagnostic run.
    recoveries=[run_dir/"resume.recovery.json"]+list((run_dir/"claim_archive").glob("**/recovery.json"))
    for recovery_path in recoveries:
        recovery=read_json(recovery_path,{})
        for item in recovery.get("worker_provenance_copies",[]):
            value=item.get("preserved_copy")
            if value:
                path=Path(value)
                path=path if path.is_absolute() else run_dir/path
                if not path.resolve().is_relative_to(run_dir):
                    raise ValueError("worker provenance escapes diagnostic run")
                if path.name.startswith("worker") and path.name.endswith("_state.json"):
                    paths.append(path)
    workers={}
    for path in sorted(set(paths)):
        if not path.resolve().is_relative_to(run_dir):
            raise ValueError("worker state escapes diagnostic run")
        worker=read_json(path,{})
        if not worker: continue
        identity=(str(worker.get("job_id")),worker.get("worker"),worker.get("pid"),worker.get("start_unix"))
        previous=workers.get(identity)
        provenance=(previous or {}).get("provenance_paths",[])+[str(path.relative_to(run_dir))]
        better=previous is None or (bool(worker.get("finished_utc")),float(worker.get("elapsed_seconds",0.)))>(bool(previous.get("finished_utc")),float(previous.get("elapsed_seconds",0.)))
        if better:
            workers[identity]=dict(worker,provenance_paths=provenance)
        else:
            previous["provenance_paths"]=provenance
    snapshots=list(workers.values())
    accounting=[]
    job_ids={str(w["job_id"]) for w in snapshots if w.get("job_id")}|{str(j) for j in extra_job_ids}
    for job_id in sorted(job_ids):
        if not job_id.isdigit():
            raise ValueError("non-numeric job identity")
        try:
            result=subprocess.run(["sacct","-n","-P","-j",job_id,
                "--format=JobIDRaw,State,ElapsedRaw,AllocTRES,ExitCode,NodeList,Start,End"],
                check=False,capture_output=True,text=True,timeout=20)
            main=[line.split("|") for line in result.stdout.splitlines() if line.split("|")[0]==job_id]
            if result.returncode or len(main)!=1:
                raise RuntimeError(f"sacct unavailable: {result.stderr.strip()}")
            values=main[0]
            record=dict(zip(("job_id","state","elapsed_seconds","alloc_tres","exit_code","node","start","end"),values))
            record["elapsed_seconds"]=int(record["elapsed_seconds"])
            tres=dict(item.split("=",1) for item in record["alloc_tres"].split(",") if "=" in item)
            gpu_count=int(tres.get("gres/gpu",sum(int(v) for k,v in tres.items() if k.startswith("gres/gpu:"))))
            record["allocated_gpus"]=gpu_count
            record["reservation_gpu_hours"]=record["elapsed_seconds"]*gpu_count/3600
            record["is_terminal"]=record["state"] not in ("RUNNING","PENDING","COMPLETING","CONFIGURING","SUSPENDED")
            accounting.append(record)
        except Exception as exc:
            accounting.append(dict(job_id=job_id,status="unavailable",error=f"{type(exc).__name__}: {exc}"))
    job_states={r["job_id"]:r for r in accounting}
    def unix_stamp(value):
        stamp=datetime.fromisoformat(value)
        if stamp.tzinfo is None:
            stamp=stamp.replace(tzinfo=ZoneInfo("America/New_York"))
        return stamp.timestamp()
    for worker in snapshots:
        job=job_states.get(str(worker.get("job_id")),{})
        start=worker.get("start_unix")
        is_archived=all(p.startswith("claim_archive/") for p in worker["provenance_paths"])
        if worker.get("finished_utc"):
            elapsed=worker.get("elapsed_seconds")
            if elapsed is None and start is not None:
                elapsed=unix_stamp(worker["finished_utc"])-float(start)
            worker["elapsed_seconds_at_report"]=float(elapsed) if elapsed is not None else None
            worker["duration_basis"]="finished_worker_state"
            worker["end_unix_for_makespan"]=unix_stamp(worker["finished_utc"])
        elif job.get("is_terminal") and job.get("end") not in (None,"Unknown") and start:
            end=unix_stamp(job["end"])
            worker["elapsed_seconds_at_report"]=max(0.,end-float(start))
            worker["duration_basis"]="upper_bound_to_terminal_Slurm_end; worker_finish_unrecorded"
            worker["end_unix_for_makespan"]=end
        elif not is_archived and job.get("is_terminal") is False and start:
            worker["elapsed_seconds_at_report"]=max(0.,time.time()-float(start))
            worker["duration_basis"]="live_worker_through_report_time"
        elif worker.get("elapsed_seconds") is not None:
            worker["elapsed_seconds_at_report"]=float(worker["elapsed_seconds"])
            worker["duration_basis"]="last_recorded_elapsed_lower_bound"
        else:
            worker["elapsed_seconds_at_report"]=None
            worker["duration_basis"]="unavailable; stale_worker_not_extrapolated_to_now"
    starts=[float(w["start_unix"]) for w in snapshots if w.get("start_unix")]
    ends=[w["end_unix_for_makespan"] for w in snapshots if w.get("end_unix_for_makespan")]
    makespan=max(ends)-min(starts) if len(ends)==len(snapshots) and starts else None
    complete_durations=all(w["elapsed_seconds_at_report"] is not None for w in snapshots)
    return dict(workers=snapshots,worker_exit=read_json(run_dir/"worker_exit.json",{}),
        parallel_worker_makespan_seconds=makespan,
        worker_process_wall_gpu_hours=sum(w["elapsed_seconds_at_report"] for w in snapshots)/3600 if snapshots and complete_durations else None,
        known_worker_process_wall_gpu_hours=sum(w["elapsed_seconds_at_report"] or 0. for w in snapshots)/3600 if snapshots else None,
        slurm=accounting,reservation_gpu_hours=sum(r["reservation_gpu_hours"] for r in accounting if "reservation_gpu_hours" in r) if accounting and all("reservation_gpu_hours" in r for r in accounting) else None,
        note="Deduplicated job/worker/process/start across archived/current snapshots. Calendar makespan includes gaps between resumed jobs. Worker wall includes non-kernel time; unfinished old workers use labelled terminal-job bounds. Reservation includes allocated idle. Neither measures utilization.")


def loadability(manifest,policy_costs,auxiliary,t2_costs):
    evidence={}
    for policy,cost in dict(policy_costs,t2_s0_prefix=auxiliary).items():
        for entry in cost.get("loads",[]):
            evidence.setdefault(str(Path(entry["path"]).resolve()),[]).append(dict(
                kind="T1_model_loaded_with_cached_FP32_head",evidence=f"policies/{policy}/costs.json",
                details=entry))
    for entry in t2_costs.get("checkpoints",[]):
        if entry.get("status")=="complete":
            evidence.setdefault(str(Path(entry["checkpoint"]).resolve()),[]).append(dict(
                kind="T2_successful_checkpoint_scoring",evidence="t2/t2_costs.json",details=entry))
    models=[("S0",manifest.get("student_base",{}))]+[(f"T{i}",r) for i,r in enumerate(manifest["route"])]
    records=[]
    for label,item in models:
        checkpoint=item.get("checkpoint")
        found=evidence.get(str(Path(checkpoint).resolve()),[]) if checkpoint else []
        records.append(dict(label=label,stage=item.get("step"),checkpoint=checkpoint,
            metadata_status=item.get("status","unknown"),actual_load_verified=bool(found),
            status="actual_load_verified" if found else "not_yet_verified_by_actual_load",
            evidence=found,weight_hash_computed=False))
    return dict(models=records,verified_count=sum(r["actual_load_verified"] for r in records),
        expected_count=len(records),note="Metadata-valid alone never establishes actual loadability. Manifest is not modified.")


def pilot_records(run_dir):
    result=[]
    for policy in ("q0_random","p5_random","q5_random"):
        pilot=read_json(run_dir/f"pilot_{policy}.json",{})
        if not pilot: continue
        cost=pilot.get("costs",{})
        elapsed=float(cost.get("generation_seconds",0.))
        result.append(dict(policy_id=policy,records=int(pilot.get("completed_records",0))-int(pilot.get("resumed_records",0)),
            generation_seconds=elapsed,generated_tokens=cost.get("generated_tokens"),
            generated_tokens_per_generation_second=cost.get("generated_tokens",0)/elapsed if elapsed else None,
            model_load_seconds=cost.get("model_load_seconds"),total_seconds=cost.get("total_wall_seconds"),
            included_in_policy_costs=True,not_added_again=True))
    return result


def t2_physical_gpu_hours(costs):
    for key in ("physical_gpu_hours","newphysical_gpu_hours","current_gpu_hours"):
        if costs.get(key) is not None:
            value=float(costs[key])
            if not math.isfinite(value) or value<0:
                raise ValueError(f"invalid T2 GPU cost {key}")
            return value,key
    return 0.,"unavailable_not_measured_zero"


ROLE_TOKEN_FIELDS=("model_scored_tokens","output_head_positions","transformer_token_positions",
                   "active_output_head_positions","useful_prefill_tokens","model_forward_calls")


def policy_role_costs(manifest,terms,cost,records):
    """Exact token-count partition only when every model shared every batch.

    This must not allocate seconds or GPU-hours by model count: a 3B and 8B
    model have different compute costs. Failures can interrupt model loops.
    """
    if not terms or not cost:
        return dict(status="unavailable",reason="no frozen terms or physical counters")
    combined={}
    for path,coefficient in terms.items():
        value=float(coefficient)
        if not math.isfinite(value):
            return dict(status="unavailable",reason="nonfinite coefficient")
        canonical=str(Path(path).resolve())
        combined[canonical]=combined.get(canonical,0.)+value
    members=[p for p,c in combined.items() if c!=0.]
    student=str(Path(manifest["student_base"]["checkpoint"]).resolve())
    teacher={str(Path(r["checkpoint"]).resolve()) for r in manifest["route"]}
    roles={"S0":0,"Teacher":0}
    for member in members:
        if member==student:
            roles["S0"]+=1
        elif member in teacher:
            roles["Teacher"]+=1
        else:
            return dict(status="unavailable",reason=f"model identity outside frozen roles: {member}")
    N=len(members)
    if not N:
        return dict(status="unavailable",reason="empty active model set")
    result=dict(status="unavailable",active_model_count=N,role_model_counts=roles,active_model_paths=members,
        derivation="Each distinct nonzero-coefficient model received the same synchronized padded batch at every successful step; divide actual all-model counts by member count, then multiply by role count.",
        time_or_gpu_hours_partitioned=False)
    if cost.get("attempts") or any(r.get("status")=="error" or r.get("error") for r in records if r.get("status")!="missing"):
        return dict(result,reason="infrastructure/model-loop failure; equal execution per member is unproven")
    per_model={}
    for key in ROLE_TOKEN_FIELDS:
        value=cost.get(key)
        if value is None or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or int(value)!=value:
            return dict(result,reason=f"missing or nonintegral actual counter: {key}")
        if int(value)%N:
            return dict(result,reason=f"actual counter not divisible by distinct model count: {key}")
        per_model[key]=int(value)//N
    return dict(result,status="derived_exact_for_recorded_synchronized_execution",per_model_counts=per_model,
        S0={k:v*roles["S0"] for k,v in per_model.items()},
        Teacher={k:v*roles["Teacher"] for k,v in per_model.items()},
        all_models={k:int(cost[k]) for k in ROLE_TOKEN_FIELDS})


def sum_role_costs(entries):
    good=[r for r in entries if r.get("status")=="derived_exact_for_recorded_synchronized_execution"]
    return dict(status="complete" if len(good)==len(entries) else "partial_known_role_counts_only",
        derived_entries=len(good),expected_entries=len(entries),
        S0={k:sum(r["S0"][k] for r in good) for k in ROLE_TOKEN_FIELDS},
        Teacher={k:sum(r["Teacher"][k] for r in good) for k in ROLE_TOKEN_FIELDS},
        all_models={k:sum(r["all_models"][k] for r in good) for k in ROLE_TOKEN_FIELDS},
        time_or_gpu_hours_partitioned=False)


def t2_role_costs(manifest,costs):
    """Use individual checkpoint records, never count reused caches as new work."""
    paths={str(Path(manifest.get("student_base",{}).get("checkpoint","unknown_S0")).resolve()):"S0"}
    paths.update({str(Path(r["checkpoint"]).resolve()):"Teacher" for r in manifest["route"] if r.get("checkpoint")})
    result=dict(status="complete",basis="successful unique checkpoint scoring in this result snapshot; reused=True contributes zero new work",
        S0=dict(lm_head_scored_positions=0,transformer_scored_tokens=0,checkpoint_count=0),
        Teacher=dict(lm_head_scored_positions=0,transformer_scored_tokens=0,checkpoint_count=0),unknown=[])
    seen=set()
    for record in costs.get("checkpoints",[]):
        if record.get("reused"):
            continue
        path=str(Path(record.get("checkpoint","unknown")).resolve())
        role=paths.get(path)
        if path in seen or role is None or record.get("status")!="complete":
            result["unknown"].append(dict(checkpoint=path,reason="duplicate, unknown role, or incomplete checkpoint record"))
            continue
        seen.add(path)
        values={k:record.get(k) for k in ("lm_head_scored_positions","transformer_scored_tokens")}
        if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 or int(v)!=v for v in values.values()):
            result["unknown"].append(dict(checkpoint=path,reason="missing or invalid actual token counters"))
            continue
        for key,value in values.items(): result[role][key]+=int(value)
        result[role]["checkpoint_count"]+=1
    if costs.get("failures") or result["unknown"] or not costs:
        result["status"]="partial_known_successful_checkpoint_counts_only"
    result["all_models"]={k:result["S0"][k]+result["Teacher"][k] for k in result["S0"]}
    if costs.get("physical_gpu_hours") is not None:
        result["cumulative_role_costs_status"]="unavailable_without_per_role_physical_attempt_ledger; current snapshot partition not claimed as cumulative"
    return result


def paired_failure_audit(records_by_policy,M):
    def category(record):
        if record["status"]=="missing": return "missing"
        if record["status"]=="error": return "infrastructure_error"
        if record["full_pass"]: return "full_pass"
        if record.get("format_error"): return "format_error"
        if record.get("parse_error"): return "parse_error"
        return "valid_parse_semantic_failure"
    result=[]
    for stage in range(M+1):
        qrows=records_by_policy[f"q{stage}_random"]
        prows={(r["row_id"],r["sample_index"]):r for r in records_by_policy[f"p{stage}_random"]}
        for q in qrows:
            p=prows[(q["row_id"],q["sample_index"])]
            missing=q["status"]=="missing" or p["status"]=="missing"
            error=q["status"]=="error" or p["status"]=="error"
            if not missing and q.get("seed")!=p.get("seed"):
                raise ValueError("paired audit seed mismatch")
            outcome=("missing" if missing else "infrastructure_contaminated" if error else
                "both_success" if q["full_pass"] and p["full_pass"] else
                "q_only_success" if q["full_pass"] else "p_only_success" if p["full_pass"] else "both_failure")
            item=dict(stage=stage,teacher_stage_name=q.get("teacher_stage_name"),row_id=q["row_id"],family=q["family"],
                sample_index=q["sample_index"],seed=q.get("seed") if q.get("seed") is not None else p.get("seed"),
                paired_outcome=outcome)
            for label,record in (("q",q),("p",p)):
                item.update({f"{label}_{name}":record.get(name) for name in
                    ("status","tier","full_pass","reward","format_error","parse_error","truncated","error")})
                item[f"{label}_failure_category"]=category(record)
                verifier=record.get("verifier") or {}
                item[f"{label}_official_error"]=verifier.get("error")
                item[f"{label}_official_valid"]=verifier.get("valid")
                item[f"{label}_official_pass_rate"]=verifier.get("pass_rate")
                item[f"{label}_official_failed_tests"]=sum(not r.get("passed",False) for r in verifier.get("results",[]))
            result.append(item)
    return result


def make_summary(manifest,profiles,comparisons,t2,costs,M,path_analysis=None):
    overall={r["policy_id"]:r for r in profiles if r["family"]=="all"}
    complete=all(r["missing"]==0 for r in overall.values())
    any_infrastructure_errors=any(r["infrastructure_errors"] for r in overall.values())
    main=next((r for r in comparisons if r["stage"]==M and r["family"]=="all" and r["mode"]=="random"),None)
    lines=["# CAFD 目标策略与真实路径诊断", "", "## 结论", ""]
    if not complete:
        lines += ["T1 尚未完成全部预定采样；下面明确保留未完成项，不能据此形成最终能力比较。"]
    elif main and main["status"]=="complete":
        q,p=overall[f"q{M}_random"],overall[f"p{M}_random"]
        lines += [f"终点相对策略 q{M} 随机完整通过 {q['correct']}/{q['N']}，对应 Teacher p{M} 为 {p['correct']}/{p['N']}。"
            f"题均成功率之差 q−p 为 {100*main['delta_success_rate']:.2f} 个百分点，"
            f"题级配对 bootstrap 95% 区间 [{100*main['success_ci_low']:.2f}, {100*main['success_ci_high']:.2f}]。"]
        if main["success_ci_high"]<0:
            lines += ["本开发诊断支持相对目标的自主生成弱于终点绝对 Teacher；这削弱了相对目标的直接设计依据，但不构成 KD Student 的能力上限或项目不可能成功的证明。"]
        elif main["success_ci_low"]>0:
            lines += ["本开发诊断支持终点相对目标的自主生成更强，但没有验证训练后的 Student 优势或蒸馏效率。"]
        else:
            lines += ["区间包含零；这批开发题不足以清楚区分终点策略的成功率。"]
    elif main and main["status"]=="contaminated":
        lines += [f"预定终点采样已记录，但 q{M} 有 {main['infrastructure_errors_q']} 条、p{M} 有 {main['infrastructure_errors_p']} 条基础设施失败。"
            "故障仍计入分母，不能把它们当成目标策略本身无能力的证据。",
            f"含故障运行的描述性差值为 {100*main['delta_success_rate']:.2f} 个百分点，"
            f"题级配对 95% 区间 [{100*main['success_ci_low']:.2f}, {100*main['success_ci_high']:.2f}]；"
            "该区间不能用于策略能力优劣结论，当前证据受故障污染。"]
    else:
        lines += ["终点策略比较仍不完整，未形成方向性结论。"]
    if any_infrastructure_errors:
        lines.append("本轮存在基础设施错误；相应策略/配对比较标为 contaminated。其数值仅描述含故障运行，不用于归因目标设计。")
    lines += ["", "## T1：自主生成，不是 Teacher replay", "",
        "随机表的 Correct/N 是通过的 completion 次数/预定次数；每题两次，题均成功率先在同题取均值。"
        "统计单位为题目，不能把 128 条 completion 当成 128 道独立题。格式错误、截断、基础设施失败均不从预定分母删除；"
        "未运行项不是已测失败，因此未齐时只报告已知通过次数，不填最终准确率。"
        "表中的平均部分奖励是对全部预定回答的原 hierarchical reward 取平均（包括 full-pass=1），不是仅对失败回答平均；"
        "基础设施失败按预定口径计零，未运行项存在时不填写最终均值。", "",
        "| 策略 | Correct/N | 已记录/预定 | 完整成功率 | 平均部分奖励 | 格式错误 | 解析错误 | 可解析语义失败 | 截断 | 基础设施错误 |", 
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for policy in planned_policies(M):
        if policy[1]!="random": continue
        r=overall[policy[0]]
        rate="未齐" if r["full_pass_rate"] is None else f"{100*r['full_pass_rate']:.2f}%"
        lines.append(f"| {r['policy_id']} | {r['correct']}/{r['N']} | {r['recorded']}/{r['N']} | {rate} | {_fmt(r['mean_partial_reward'])} | {r['format_errors']} | {r['parse_errors']} | {r['valid_parse_semantic_failures']} | {r['truncations']} | {r['infrastructure_errors']} |")
    lines += ["", "Greedy 单独报告，每题一次：", "", "| 策略 | Correct/N | 成功率 | 已记录 |", "|---|---:|---:|---:|"]
    for policy in planned_policies(M):
        if policy[1]!="greedy": continue
        r=overall[policy[0]]
        rate="未齐" if r["full_pass_rate"] is None else f"{100*r['full_pass_rate']:.2f}%"
        lines.append(f"| {r['policy_id']} | {r['correct']}/{r['N']} | {rate} | {r['recorded']} |")
    lines += ["", "总体及分题型详见 `target_profile.csv`；每阶段 q_m−p_Tm 的题级配对区间见 `paired_comparisons.csv`。"
        "随机与 greedy 不混算。q0 直接复用 S0。目标成功率曲线平坦或非单调，单独不能证明课程无效。", "",
        "可解析语义失败只计 status=ok、未完整通过、格式正确且可解析的条目。截断是单独标记，可能与失败类型交叉；"
        "profile 同时列出其中截断/未截断数量。逐题同 seed 的 q/p 错误层级、官方解析错误、测试失败数与 Teacher 成功情况见"
        " `paired_failure_audit.jsonl` / `paired_failure_audit.csv`。", "",
        "## T2：真实路径与终点插值", ""]
    endpoint_audit=costs.get("endpoint_failure_audit",{})
    if complete and main and main["status"]=="complete" and endpoint_audit:
        insertion=(f"终点错误层级：q{M} 解析错误 {endpoint_audit['q_parse_errors']} 条，p{M} {endpoint_audit['p_parse_errors']} 条；"
            f"可解析语义失败分别为 {endpoint_audit['q_valid_parse_semantic_failures']} 与 {endpoint_audit['p_valid_parse_semantic_failures']} 条。"
            f"相同题目与 seed 下，q 解析失败而 p 成功 {endpoint_audit['q_parse_p_success']} 条，反向 {endpoint_audit['p_parse_q_success']} 条。"
            "这些是失败类型和配对输出的描述性关联；若通过率差距与解析错误伴随出现，仍不能将其直接归因为 logit 相减机制。"
            "两条策略随后生成的前缀不同，未实施保持前缀不变的机制因果干预。")
        if endpoint_audit["q_parse_errors"]:
            insertion+=(f" q 的解析错误涉及 {endpoint_audit['q_parse_question_count']} 道不同题目，题型计数为 "
                +json.dumps(endpoint_audit["q_parse_family_counts"],ensure_ascii=False)+"；官方错误文本计数为 "
                +json.dumps(endpoint_audit["q_parse_official_error_counts"],ensure_ascii=False)+
                "。这里区分的是语法解析失败，不能与格式错误或截断混称。")
        t2_index=lines.index("## T2：真实路径与终点插值")
        lines[t2_index:t2_index]=[insertion,""]
    if not t2:
        lines.append("T2 几何结果尚未生成。")
    else:
        lines += [f"状态：{t2.get('status','unknown')}。精确计算 KL(q_m || q_endpoint(a))，全词表统一归一化。"
            "每阶段只拟合一个跨来源共享的标量 a，且只使用校准题；按题分开的留出题用于报告残差。", "",
            "| 来源 | m | 固定 a | 拟合 a | 固定校准 KL | 拟合校准 KL | 固定留出 KL | 拟合留出 KL | 留出题数 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in t2.get("stage_results",[]):
            lines.append(f"| {r['source']} | {r['stage']} | {_fmt(r['fixed_a'])} | {_fmt(r['fitted_a'])} | {_fmt(r.get('fixed_calibration_kl'))} | {_fmt(r.get('fitted_calibration_kl'))} | {_fmt(r.get('fixed_heldout_kl'))} | {_fmt(r.get('fitted_heldout_kl'))} | {r.get('heldout_questions','')} |")
        if t2.get("status")!="complete" or t2.get("failures"):
            lines.append("\nT2 为部分完成或包含失败，不能对完整路线是否等价、是否有序作确定结论。"
                "上表仅描述实际可用节点；已观察到的系数回退记录为："+json.dumps(t2.get("coefficient_reversals",[]),ensure_ascii=False)+"。")
        elif not t2.get("ordering_complete_route"):
            lines.append("\n路线有缺项，不能判定完整路线的拟合系数顺序。")
        elif t2.get("coefficient_reversals"):
            lines.append("\n拟合系数出现回退："+json.dumps(t2["coefficient_reversals"],ensure_ascii=False)+"。因此不满足‘有序变速’解释的顺序条件。")
        else:
            lines.append("\n拟合系数未检测到超过 1e-6 的回退；这只满足顺序条件。未预设残差等价阈值，不能由此宣称真实路径等价于终点插值。")
        sources=sorted({r["source"] for r in t2.get("stage_results",[]) if r["source"]!="all"})
        lines.append("\n本轮可用前缀来源："+", ".join(sources)+"。不同来源残差分别列示；S0 的结果不能替代早期/晚期学生前缀结果。"
            "早期/晚期是历史 Student 自身生成的前缀，解码可能与本轮 T1 不同；没有用 gold 前缀替代。")
        source_means=[]
        for source in sources:
            interior=[r for r in t2.get("stage_results",[]) if r["source"]==source and 0<int(r["stage"])<M
                      and r.get("fixed_heldout_kl") is not None and r.get("fitted_heldout_kl") is not None]
            if interior:
                fixed=float(np.mean([r["fixed_heldout_kl"] for r in interior]))
                fitted=float(np.mean([r["fitted_heldout_kl"] for r in interior]))
                source_means.append((source,fixed,fitted,len(interior)))
        if source_means:
            lines += ["", "跨前缀来源检查（可用中间阶段的等权算术平均留出 KL，不包含恒等端点）：", "",
                "| 来源 | 阶段数 | 固定 a 残差 | 拟合 a 残差 |", "|---|---:|---:|---:|"]
            for source,fixed,fitted,n in source_means:
                lines.append(f"| {source} | {n} | {fixed:.6f} | {fitted:.6f} |")
            directions={fitted<fixed for _,fixed,fitted,_ in source_means}
            lines.append("\n"+("各来源的平均留出残差均下降。" if directions=={True} else
                "各来源并非都得到平均留出残差下降；不能把校准优化等同于跨来源的改善。")+
                "这里的 a 是跨来源共享拟合，而不是各来源分别挑选；该比较不能证明来源分布等价或 Student 会学到相同能力。")
        if t2.get("failures"):
            lines.append("\nT2 限制/失败：`t2/path_geometry.json` 的 failures 已完整保留；缺失模型未重训或替换。")
        if path_analysis:
            excerpt=path_analysis.split("## 结论先行",1)[-1].split("## 1.",1)[0].strip()
            lines += ["", "### 路径差异的科学解释", "",
                "详细解释与逐阶段证据见 [path_analysis.md](path_analysis.md)。"
                "需要区分首段在几何上已接近终点、后续小幅系数回退、不同 Student 前缀来源的残差差异。"
                "小幅回退不能自动称作强烈的课程逆转；拟合系数受限于 [0,1] 时，边界上的剩余 KL 也可能反映沿终点方向的超调，"
                "不必然证明出现新的正交知识方向。",
                excerpt]
    selection=manifest.get("selection",{})
    lines += ["", "## 数据范围与解释边界", "",
        f"Benchmark：{manifest.get('benchmark','见 manifest')}。Student 为原始 Ministral 3B S0，Teacher 为 Ministral 8B；capacity SFT Student 未冒充 S0。",
        f"使用已命名 selection 64 题：Teacher SFT 重叠 {selection.get('teacher_sft_overlap_count','待核实')}/64，"
        f"Teacher RL 重叠 {selection.get('teacher_rl_overlap_count','待核实')}/64。该集合也有历史 Student 选择暴露，不能称作独立泛化证据。"
        "Confirmation 也有历史选择/观察暴露，本轮未使用；未读取 frozen-test 结果。",
        "Teacher 路线保留真实原始 Instruct → SFT125 → RL40 → RL60 → RL80 → RL100 顺序。"
        "T2 是有历史前缀覆盖的固定 fit 题子集，不代表总体无偏抽样。",
        "目标策略的成功率不是 KD Student 的硬天花板；q_M 是否达到 52/64 不能单独决定 Student 的可达性。"
        "目标间 KL 与历史拟合误差若前缀、方向、归一化不一致不能比较，更不能据此断言学生无法分辨。", "",
        "真实节点及实际加载核验（metadata-valid 不替代实加载）：", "",
        "| 节点 | 真实训练阶段 | 实际加载 | checkpoint |", "|---|---|---|---|"]
    for model in costs.get("verified_loadability",{}).get("models",[]):
        path=model.get("checkpoint") or "缺失"
        lines.append(f"| {model['label']} | {model.get('stage','未知')} | {'已验证' if model['actual_load_verified'] else '尚无成功加载证据'} | `{path}` |")
    lines += ["", "## 成本与下一轮", "",
        f"T1 已记录生成 tokens：{int(costs['t1']['generated_tokens'])}；T2 补前缀生成 tokens：{int(costs['auxiliary_prefix']['generated_tokens'])}。"
        f"已记录 GPU 阶段时间总和：{costs['recorded_gpu_stage_hours']:.4f} GPU-hours。"
        "该值是各 GPU 工作阶段累加，不是双卡并行总墙钟；未覆盖的预约空闲/故障重试必须另列，不能当作零。",
        "详细模型加载、prefill/decode、实际 head 评分（含 padding/已停止行空耗）、verifier、T2 缓存与显存见 `costs.json`。",
        "已准备累计相对目标与同阶段绝对目标两条纯 KD 配置：共用固定五阶段日程、同一 S0、200 updates、"
        "seed 2027、当前 Student 前缀和 exact full-vocabulary forward KL。没有 RL loss、奖励路由、MPC 或自适应调度；本轮未提交训练。"]
    if costs.get("pilot_throughput"):
        lines += ["", "先行小批次实测（这些成本已计入对应策略，不重复累加）：", "",
            "| 策略 | completion数 | 纯生成秒 | 生成tokens | tokens/生成秒 | 含加载总秒 |",
            "|---|---:|---:|---:|---:|---:|"]
        for p in costs["pilot_throughput"]:
            lines.append(f"| {p['policy_id']} | {p['records']} | {_fmt(p['generation_seconds'],3)} | {p['generated_tokens']} | {_fmt(p['generated_tokens_per_generation_second'],2)} | {_fmt(p['total_seconds'],3)} |")
        lines.append("\n小批次长度与后续策略可能不同，不能用首批秒数保证完成时限；终态以实际记录成本为准。")
    role_accounting=costs.get("model_role_accounting",{})
    if role_accounting:
        lines += ["", "生成与评分计数（不能把所有模型的评分量都称为 Teacher tokens）：", "",
            f"全部诊断共生成 {int(costs['generated_tokens_all_diagnostics'])} tokens，包含 T1 与辅助 32 条 S0 前缀，"
            "每个输出 token 只计一次；pilot 已在对应策略中计入，不重复累加。", "",
            "| 阶段 | Teacher 输出头位置 | S0 输出头位置 | 总模型输出头位置 | Teacher Transformer输入位置 | S0 Transformer输入位置 |",
            "|---|---:|---:|---:|---:|---:|"]
        for label,key in (("T1 已知可分离计数","T1"),("辅助 S0 前缀","auxiliary_S0_prefix"),("T2 本次新增评分","T2")):
            value=role_accounting[key]
            if "S0" not in value or "Teacher" not in value:
                lines.append(f"| {label} | 无法可靠分离 | 无法可靠分离 | 见原始计数 | 无法可靠分离 | 无法可靠分离 |")
                continue
            head="lm_head_scored_positions" if key=="T2" else "output_head_positions"
            transformer="transformer_scored_tokens" if key=="T2" else "transformer_token_positions"
            teacher,student=value["Teacher"],value["S0"]
            lines.append(f"| {label} | {teacher[head]} | {student[head]} | {teacher[head]+student[head]} | {teacher[transformer]} | {student[transformer]} |")
        lines.append("\nT1 的实际 head/Transformer 位置包含 padding 与已停止行的空耗。角色分离仅在冻结 terms 去重后的各模型同步执行、"
            "无中途基础设施错误且计数整除时成立；否则保留未知，不按比例猜测。模型大小不同，因此没有把时间或 GPU-hours 按模型数量平摊。"
            "T2 若在未来恢复中复用缓存，表中仅计本次新增评分，累计物理成本另由账本记录，不能混算。")
    execution=costs.get("execution",{})
    if execution.get("reservation_gpu_hours") is not None:
        terminal=bool(execution.get("slurm")) and all(j.get("is_terminal") for j in execution["slurm"])
        timing_note="本次作业已结束，上述为终态记录；" if terminal else "运行中数值不是终态成本；"
        lines.append(f"Slurm 记录的预约资源为 {execution['reservation_gpu_hours']:.4f} GPU-hours；"
            f"worker 进程墙钟和为 {_fmt(execution.get('worker_process_wall_gpu_hours'))} GPU-hours。"
            f"双 worker 首起到末终的实际墙钟为 {_fmt(execution.get('parallel_worker_makespan_seconds'),1)} 秒。"
            +timing_note+"详细 job/node/exit 和查询状态见 execution。")
    if not complete or any_infrastructure_errors or not t2 or t2.get("status")!="complete":
        lines.append("当前诊断存在未完成项或故障污染，证据不足以给出完整投资判断；继续完成可运行诊断，不用缺项或基础设施错误推断方法负结果。")
    elif main and main["success_ci_high"]<0:
        lines.append("若下一轮继续，定位应是有界预算的目标假设对照，而不是已获证实的提升验证。"
            "相对目标自主生成偏弱是风险信号；最关键的不确定性仍是同预算 KD Student 的实际学习响应。")
    else:
        lines.append("下一轮两条固定调度训练能够直接检验目标选择对 Student 学习的影响；本轮尚未证明相对目标更有效。"
            "主要不确定性是目标几何与自主生成差异能否转化为同预算训练收益，以及在新任务上的泛化。")
    return "\n".join(lines)+"\n"


def aggregate(run_dir):
    run_dir=Path(run_dir)
    manifest=read_json(run_dir/"manifest.json")
    selection=read_jsonl(run_dir/"selection.jsonl")
    if not manifest or not selection:
        raise ValueError("frozen manifest and selection required")
    if len({str(r["id"]) for r in selection})!=len(selection):
        raise ValueError("duplicate selection IDs")
    if manifest.get("selection",{}).get("ids") != [str(r["id"]) for r in selection]:
        raise ValueError("selection order/IDs differ from frozen manifest")
    M=len(manifest["route"])-1
    policies=planned_policies(M)
    records_by_policy,all_records,profiles,policy_costs={ },[],[],{}
    for policy in policies:
        records=collect_policy(run_dir,policy,selection)
        cost=read_json(run_dir/"policies"/policy[0]/"costs.json",{})
        identity=read_json(run_dir/"policies"/policy[0]/"identity.json",{})
        cost=dict(cost,role_breakdown=policy_role_costs(manifest,identity.get("terms"),cost,records))
        records_by_policy[policy[0]]=records
        all_records.extend(records)
        profiles.extend(profile(records,policy,cost))
        policy_costs[policy[0]]=cost
    comparisons=[]
    families=["all"]+sorted({r["problem_family"] for r in selection})
    for stage in range(M+1):
        for family in families:
            qs=records_by_policy[f"q{stage}_random"]
            ps=records_by_policy[f"p{stage}_random"]
            if family!="all":
                qs=[r for r in qs if r["family"]==family]
                ps=[r for r in ps if r["family"]==family]
            item=paired_bootstrap(qs,ps)
            comparisons.append(dict(stage=stage,family=family,mode="random",contrast="q_m - p_Tm",**item))
    for row in all_records+profiles+comparisons:
        row["teacher_stage_name"]=manifest["route"][int(row["stage"])].get("step",f"T{row['stage']}")
    for row in profiles:
        role=policy_costs[row["policy_id"]]["role_breakdown"]
        known=role.get("status")=="derived_exact_for_recorded_synchronized_execution" and row["family"]=="all"
        row["role_count_derivation_status"]=role["status"] if row["family"]=="all" else None
        row["teacher_model_scored_tokens"]=role["Teacher"]["model_scored_tokens"] if known else None
        row["S0_model_scored_tokens"]=role["S0"]["model_scored_tokens"] if known else None
        row["teacher_transformer_token_positions"]=role["Teacher"]["transformer_token_positions"] if known else None
        row["S0_transformer_token_positions"]=role["S0"]["transformer_token_positions"] if known else None
    audits=paired_failure_audit(records_by_policy,M)
    with (run_dir/"target_rollouts.jsonl").open("w",encoding="utf-8") as handle:
        for record in all_records:
            handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")
    write_csv(run_dir/"target_profile.csv",profiles)
    write_csv(run_dir/"paired_comparisons.csv",comparisons)
    write_csv(run_dir/"paired_failure_audit.csv",audits)
    with (run_dir/"paired_failure_audit.jsonl").open("w",encoding="utf-8") as handle:
        for item in audits:
            handle.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+"\n")
    t2=read_json(run_dir/"t2/path_geometry.json",{})
    if (run_dir/"t2/path_geometry.csv").is_file():
        (run_dir/"path_geometry.csv").write_bytes((run_dir/"t2/path_geometry.csv").read_bytes())
    else:
        write_csv(run_dir/"path_geometry.csv",[])
    t2_costs=read_json(run_dir/"t2/t2_costs.json",{})
    auxiliary=read_json(run_dir/"policies/t2_s0_prefix/costs.json",{})
    auxiliary_identity=read_json(run_dir/"policies/t2_s0_prefix/identity.json",{})
    auxiliary_records=read_jsonl(run_dir/"policies/t2_s0_prefix/rollouts.jsonl")
    auxiliary_role=policy_role_costs(manifest,auxiliary_identity.get("terms"),auxiliary,auxiliary_records)
    t1_roles=sum_role_costs([c["role_breakdown"] for c in policy_costs.values()])
    t2_roles=t2_role_costs(manifest,t2_costs)
    t1_cost=_sum_costs(list(policy_costs.values()))
    aux_cost=_sum_costs([auxiliary])
    t2_gpu_hours,t2_gpu_basis=t2_physical_gpu_hours(t2_costs)
    total_gpu=(t1_cost["gpu_wall_seconds"]+aux_cost["gpu_wall_seconds"])/3600+t2_gpu_hours
    recorded_complete=all(r["status"]!="missing" for r in all_records) and t2.get("status")=="complete"
    infrastructure_errors=any(r["status"]=="error" for r in all_records)
    costs=dict(status=("contaminated" if infrastructure_errors else "complete") if recorded_complete else "incomplete",
        generated_utc=datetime.now(timezone.utc).isoformat(),t1=t1_cost,t1_by_policy=policy_costs,
        auxiliary_prefix=aux_cost,auxiliary_prefix_detail=auxiliary,t2=t2_costs,
        model_role_accounting=dict(T1=t1_roles,auxiliary_S0_prefix=auxiliary_role,T2=t2_roles,
            note="Teacher and S0 are partitioned by actual deduplicated synchronized model calls, never weighted coefficients. T2 uses per-checkpoint scoring counters. No proportional GPU-time allocation."),
        t2_gpu_hours_counted=t2_gpu_hours,t2_gpu_hours_basis=t2_gpu_basis,
        recorded_gpu_stage_hours=total_gpu,
        total_parallel_wall_seconds=None,
        definitions=dict(bootstrap="10,000 paired question resamples; seed=2027; per-question two-completion means",
            t1_correct_denominator="128 scheduled completions (64 questions x 2), errors count zero, missing remains unknown",
            gpu_stage_hours="sum of recorded one-GPU work intervals; not Slurm allocation walltime or parallel makespan",
            t1_model_scored_tokens="actual output-head positions including padded inactive batch rows",
            t2_transformer_scored_tokens="all Transformer sequence positions used to compute predeclared prefix logits",
            unknown_overhead="driver setup, allocation idle time, unpersisted killed attempts not silently set to zero"),
        expected_t1_episodes=len(all_records),recorded_t1_episodes=sum(r["status"]!="missing" for r in all_records),
        failed_t1_episodes=sum(r["status"]=="error" for r in all_records),
        teacher_training_included=False,optimizer_updates=0,frozen_test_read=False)
    costs["generated_tokens_all_diagnostics"]=t1_cost["generated_tokens"]+aux_cost["generated_tokens"]+t2_costs.get("generated_tokens",0)
    costs["generated_tokens_all_diagnostics_definition"]="Every output token counted once: T1 policies plus the 32 auxiliary S0 prefixes plus any explicitly recorded T2 generation. Pilot is already included in T1, not added again; model multiplicity never multiplies generated tokens."
    costs["execution"]=execution_accounting(run_dir,t2_costs.get("physical_job_ids",[]))
    costs["total_parallel_wall_seconds"]=costs["execution"]["parallel_worker_makespan_seconds"]
    costs["pilot_throughput"]=pilot_records(run_dir)
    endpoint=[a for a in audits if a["stage"]==M]
    costs["endpoint_failure_audit"]=dict(
        q_parse_errors=sum(a["q_failure_category"]=="parse_error" for a in endpoint),
        p_parse_errors=sum(a["p_failure_category"]=="parse_error" for a in endpoint),
        q_valid_parse_semantic_failures=sum(a["q_failure_category"]=="valid_parse_semantic_failure" for a in endpoint),
        p_valid_parse_semantic_failures=sum(a["p_failure_category"]=="valid_parse_semantic_failure" for a in endpoint),
        q_parse_p_success=sum(a["q_failure_category"]=="parse_error" and a["p_full_pass"] for a in endpoint),
        p_parse_q_success=sum(a["p_failure_category"]=="parse_error" and a["q_full_pass"] for a in endpoint),
        q_parse_question_count=len({a["row_id"] for a in endpoint if a["q_failure_category"]=="parse_error"}),
        q_parse_family_counts=dict(Counter(a["family"] for a in endpoint if a["q_failure_category"]=="parse_error")),
        q_parse_official_error_counts=dict(Counter(a["q_official_error"] or "not_recorded" for a in endpoint if a["q_failure_category"]=="parse_error")),
        missing_pairs=sum(a["paired_outcome"]=="missing" for a in endpoint),
        infrastructure_contaminated_pairs=sum(a["paired_outcome"]=="infrastructure_contaminated" for a in endpoint),
        causal_claim=False)
    costs["verified_loadability"]=loadability(manifest,policy_costs,auxiliary,t2_costs)
    write_json(run_dir/"verified_loadability.json",costs["verified_loadability"])
    write_json(run_dir/"costs.json",costs)
    analysis=(run_dir/"path_analysis.md").read_text(encoding="utf-8") if (run_dir/"path_analysis.md").is_file() else None
    (run_dir/"summary.md").write_text(make_summary(manifest,profiles,comparisons,t2,costs,M,analysis),encoding="utf-8")
    return costs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    args=parser.parse_args()
    result=aggregate(args.run_dir)
    print(json.dumps({k:result[k] for k in ("status","expected_t1_episodes","recorded_t1_episodes","failed_t1_episodes")}),flush=True)


if __name__=="__main__":
    main()
