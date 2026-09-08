import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from cafd.target_path_report import (aggregate,collect_policy,paired_bootstrap,
                                    planned_policies,profile,execution_accounting,t2_physical_gpu_hours,
                                    paired_failure_audit,policy_role_costs,t2_role_costs,ROLE_TOKEN_FIELDS)


def episode(policy,task,replicate,success,status="ok"):
    return dict(policy_id=policy,row_id=task,sample_index=replicate,seed=2027+replicate,
        family="family",sample=not policy.endswith("greedy"),status=status,
        error="failure" if status=="error" else None,full_pass=success,
        reward=float(success),generated_tokens=1,completion_ids=[2],completion_text="",
        truncated=False,format_error=False,parse_error=False)


def test_question_not_completion_bootstrap_has_expected_variance():
    # Question a has two successes, b has two failures. True n=2, never n=4.
    q=[episode("q","a",j,True) for j in range(2)]+[episode("q","b",j,False) for j in range(2)]
    p=[episode("p",task,j,False) for task in ("a","b") for j in range(2)]
    result=paired_bootstrap(q,p,draws=10000,seed=2027)
    assert result["n_questions"]==2
    assert result["delta_success_rate"]==.5
    assert result["success_ci_low"]==0 and result["success_ci_high"]==1
    assert result==paired_bootstrap(q,p,draws=10000,seed=2027)


def test_within_question_opposite_completions_zero_bootstrap_variance():
    q=[episode("q",task,j,j==0) for task in ("a","b") for j in range(2)]
    p=[episode("p",task,j,j==1) for task in ("a","b") for j in range(2)]
    result=paired_bootstrap(q,p)
    assert result["delta_success_rate"]==result["success_ci_low"]==result["success_ci_high"]==0


def test_seed_mismatch_refused_and_missing_withholds_final_ci():
    q=[episode("q","a",j,False) for j in range(2)]
    p=[episode("p","a",j,False) for j in range(2)]
    p[0]["seed"]=9
    with pytest.raises(ValueError,match="seed"):
        paired_bootstrap(q,p)
    p[0]["status"]="missing"
    assert paired_bootstrap(q,p)["success_ci_low"] is None


def fixture_run(tmp_path):
    rows=[dict(id="a",problem_family="family"),dict(id="b",problem_family="family")]
    (tmp_path/"selection.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    (tmp_path/"manifest.json").write_text(json.dumps(dict(route=[{},{}],selection=dict(ids=["a","b"]))))
    return rows


def save_policy(tmp_path,policy,rows):
    p=tmp_path/"policies"/policy
    p.mkdir(parents=True,exist_ok=True)
    (p/"rollouts.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")


def test_collect_missing_never_silently_changes_denominator(tmp_path):
    selection=fixture_run(tmp_path)
    save_policy(tmp_path,"q0_random",[episode("q0_random","a",0,True)])
    records=collect_policy(tmp_path,("q0_random","random",2,0,"q"),selection)
    assert len(records)==4 and sum(r["status"]=="missing" for r in records)==3
    report=profile(records,("q0_random","random",2,0,"q"),{})[0]
    assert report["N"]==4 and report["correct"]==1
    assert report["full_pass_rate"] is None
    assert report["full_pass_rate_known_lower_bound"]==.25
    assert report["full_pass_rate_unknown_upper_bound"]==1.


def test_duplicate_and_wrong_task_refused(tmp_path):
    selection=fixture_run(tmp_path)
    item=episode("q0_random","a",0,True)
    save_policy(tmp_path,"q0_random",[item,item])
    with pytest.raises(ValueError,match="duplicate"):
        collect_policy(tmp_path,("q0_random","random",2,0,"q"),selection)
    save_policy(tmp_path,"q0_random",[episode("q0_random","unexpected",0,False)])
    with pytest.raises(ValueError,match="unexpected"):
        collect_policy(tmp_path,("q0_random","random",2,0,"q"),selection)


def test_errors_kept_as_zero_but_not_missing(tmp_path):
    selection=fixture_run(tmp_path)
    records=[episode("q0_random",task,j,False,status="error") for task in ("a","b") for j in range(2)]
    save_policy(tmp_path,"q0_random",records)
    actual=collect_policy(tmp_path,("q0_random","random",2,0,"q"),selection)
    r=profile(actual,("q0_random","random",2,0,"q"),{})[0]
    assert r["status"]=="contaminated" and r["N"]==4
    assert r["infrastructure_errors"]==4 and r["full_pass_rate"]==0


def test_full_incomplete_aggregate_does_not_read_test_or_auxiliary_as_t1(tmp_path):
    fixture_run(tmp_path)
    save_policy(tmp_path,"t2_s0_prefix",[episode("t2_s0_prefix","not_selection",0,False)])
    (tmp_path/"policies/t2_s0_prefix/costs.json").write_text(json.dumps(dict(generated_tokens=100,gpu_wall_seconds=60)))
    (tmp_path/"frozen_test.json").write_text("deliberately invalid JSON; must never be accessed")
    result=aggregate(tmp_path)
    assert result["status"]=="incomplete"
    assert result["expected_t1_episodes"]==22 and result["recorded_t1_episodes"]==0
    assert result["auxiliary_prefix"]["generated_tokens"]==100 and result["t1"]["generated_tokens"]==0
    assert result["generated_tokens_all_diagnostics"]==100
    outputs=[json.loads(x) for x in (tmp_path/"target_rollouts.jsonl").read_text().splitlines()]
    assert len(outputs)==22 and all(r["status"]=="missing" for r in outputs)
    text=(tmp_path/"summary.md").read_text()
    assert "硬天花板" in text and "T1 尚未完成" in text
    assert "不是仅对失败回答平均" in text


def test_protocol_five_route_has_1728_scheduled_episodes():
    assert sum(count*64 for _,_,count,_,_ in planned_policies(5))==1728


def test_infrastructure_failure_ci_is_descriptive_not_capability_conclusion():
    q=[episode("q",task,j,False,status="error") for task in ("a","b") for j in range(2)]
    p=[episode("p",task,j,True) for task in ("a","b") for j in range(2)]
    result=paired_bootstrap(q,p)
    assert result["status"]=="contaminated"
    assert result["infrastructure_errors_q"]==4 and result["infrastructure_errors_p"]==0
    assert result["delta_success_rate"]==-1. and result["success_ci_high"]==-1.


def test_complete_but_infra_errors_never_claim_target_weaker(tmp_path):
    fixture_run(tmp_path)
    for policy,mode,count,stage,kind in planned_policies(1):
        failed=policy=="q1_random"
        rows=[episode(policy,task,j,not failed,status="error" if failed else "ok")
              for task in ("a","b") for j in range(count)]
        save_policy(tmp_path,policy,rows)
    (tmp_path/"t2").mkdir()
    (tmp_path/"t2/path_geometry.json").write_text(json.dumps(dict(status="complete",stage_results=[],
        failures=[],ordering_complete_route=True,coefficient_reversals=[])))
    result=aggregate(tmp_path)
    assert result["status"]=="contaminated"
    text=(tmp_path/"summary.md").read_text()
    assert "当前证据受故障污染" in text
    assert "本开发诊断支持相对目标的自主生成弱于" not in text


def test_partial_t2_never_claims_complete_ordering(tmp_path):
    fixture_run(tmp_path)
    (tmp_path/"t2").mkdir()
    (tmp_path/"t2/path_geometry.json").write_text(json.dumps(dict(status="partial",stage_results=[],
        failures=[dict(error="missing checkpoint")],ordering_complete_route=True,coefficient_reversals=[])))
    aggregate(tmp_path)
    text=(tmp_path/"summary.md").read_text()
    assert "不能对完整路线是否等价、是否有序作确定结论" in text
    assert "拟合系数未检测到超过" not in text


def test_physical_t2_ledger_prioritized_over_last_attempt_snapshot():
    assert t2_physical_gpu_hours(dict(physical_gpu_hours=3.,current_gpu_hours=.25))==(3.,"physical_gpu_hours")
    assert t2_physical_gpu_hours(dict(newphysical_gpu_hours=2.,current_gpu_hours=.25))==(2.,"newphysical_gpu_hours")
    assert t2_physical_gpu_hours(dict(current_gpu_hours=.25))==(.25,"current_gpu_hours")
    with pytest.raises(ValueError):
        t2_physical_gpu_hours(dict(physical_gpu_hours=-1.))


def test_archived_workers_and_slurm_jobs_deduplicated_without_stale_clock_growth(tmp_path):
    archived=tmp_path/"claim_archive/before_20/worker_provenance"
    archived.mkdir(parents=True)
    old=dict(job_id="10",worker=0,pid=1,start_unix=100,status="running")
    (archived/"worker0_state.json").write_text(json.dumps(old))
    duplicate=tmp_path/"claim_archive/another_copy/worker_provenance"
    duplicate.mkdir(parents=True)
    (duplicate/"worker0_state.json").write_text(json.dumps(old))
    new=dict(job_id="20",worker=0,pid=2,start_unix=300,status="complete",elapsed_seconds=100,
             finished_utc="1970-01-01T00:06:40+00:00")
    (tmp_path/"worker0_state.json").write_text(json.dumps(new))
    (tmp_path/"resume.recovery.json").write_text(json.dumps(dict(worker_provenance_copies=[
        dict(preserved_copy=str(archived/"worker0_state.json"))])))
    calls=[]
    def sacct(args,**kwargs):
        job=args[args.index("-j")+1]
        calls.append(job)
        end="1970-01-01T00:03:20+00:00" if job=="10" else "1970-01-01T00:06:40+00:00"
        return SimpleNamespace(returncode=0,stderr="",stdout=f"{job}|COMPLETED|110|gres/gpu:b200=2,gres/gpu=2|0:0|node|Unknown|{end}|\n")
    with patch("cafd.target_path_report.subprocess.run",side_effect=sacct):
        result=execution_accounting(tmp_path,extra_job_ids=["10"])
    assert calls==["10","20"]
    assert len(result["workers"])==2
    assert result["worker_process_wall_gpu_hours"]==200/3600
    assert result["parallel_worker_makespan_seconds"]==300
    assert result["reservation_gpu_hours"]==440/3600
    prior=next(w for w in result["workers"] if w["job_id"]=="10")
    assert prior["duration_basis"].startswith("upper_bound")
    assert len(prior["provenance_paths"])==2


def test_failure_audit_distinguishes_format_parse_semantic_truncation_and_infra():
    q=[episode("q0_random",task,j,False) for task in ("a","b") for j in range(2)]
    p=[episode("p0_random",task,j,True) for task in ("a","b") for j in range(2)]
    q[0]["format_error"]=True
    q[1]["parse_error"]=True
    q[1]["verifier"]=dict(error="Invalid node declaration: PULLER end",valid=False,results=[])
    q[2]["truncated"]=True
    q[3]["status"]="error"
    q[3]["error"]="OOM"
    result=profile(q,("q0_random","random",2,0,"q"),{})[0]
    assert result["format_errors"]==1 and result["parse_errors"]==1
    assert result["valid_parse_semantic_failures"]==1
    assert result["valid_parse_semantic_failures_truncated"]==1
    assert result["infrastructure_errors"]==1
    audit=paired_failure_audit(dict(q0_random=q,p0_random=p),0)
    assert [r["q_failure_category"] for r in audit]==[
        "format_error","parse_error","valid_parse_semantic_failure","infrastructure_error"]
    assert audit[1]["q_official_error"]=="Invalid node declaration: PULLER end"
    assert audit[1]["p_full_pass"] is True
    assert audit[3]["paired_outcome"]=="infrastructure_contaminated"


def test_t1_exact_role_partition_by_deduplicated_terms_without_time_guessing(tmp_path):
    s,t0,t1=[str(tmp_path/name) for name in ("S0","T0","T1")]
    manifest=dict(student_base=dict(checkpoint=s),route=[dict(checkpoint=t0),dict(checkpoint=t1)])
    cost={key:300 for key in ROLE_TOKEN_FIELDS}
    cost.update(gpu_wall_seconds=99,attempts=[])
    result=policy_role_costs(manifest,{s:1.,t0:-1.,t1:1.},cost,[])
    assert result["role_model_counts"]==dict(S0=1,Teacher=2)
    assert result["S0"]["model_scored_tokens"]==100
    assert result["Teacher"]["transformer_token_positions"]==200
    assert result["all_models"]["output_head_positions"]==300
    assert result["time_or_gpu_hours_partitioned"] is False
    assert "gpu_wall_seconds" not in result["Teacher"]
    # Zero coefficient T0 must not increase the member count for q0.
    zero=policy_role_costs(manifest,{s:1.,t0:0.},cost,[])
    assert zero["role_model_counts"]==dict(S0=1,Teacher=0)
    assert zero["S0"]["output_head_positions"]==300


def test_t1_role_partition_refuses_nondivisible_or_interrupted_counts(tmp_path):
    s,t0,t1=[str(tmp_path/name) for name in ("S0","T0","T1")]
    manifest=dict(student_base=dict(checkpoint=s),route=[dict(checkpoint=t0),dict(checkpoint=t1)])
    cost={key:300 for key in ROLE_TOKEN_FIELDS}
    bad=policy_role_costs(manifest,{s:1.,t0:-1.,t1:1.},dict(cost,model_scored_tokens=301),[])
    assert bad["status"]=="unavailable" and "divisible" in bad["reason"]
    error=policy_role_costs(manifest,{s:1.,t0:-1.,t1:1.},cost,[dict(status="error",error="OOM")])
    assert error["status"]=="unavailable" and "S0" not in error


def test_t2_role_counters_use_six_teachers_and_exclude_reuse(tmp_path):
    s=str(tmp_path/"S0")
    teachers=[str(tmp_path/f"T{i}") for i in range(6)]
    manifest=dict(student_base=dict(checkpoint=s),route=[dict(checkpoint=p) for p in teachers])
    checkpoints=[dict(checkpoint=p,status="complete",reused=False,lm_head_scored_positions=1536,
        transformer_scored_tokens=120726) for p in [s]+teachers]
    result=t2_role_costs(manifest,dict(checkpoints=checkpoints))
    assert result["Teacher"]["lm_head_scored_positions"]==9216
    assert result["Teacher"]["transformer_scored_tokens"]==724356
    assert result["S0"]["lm_head_scored_positions"]==1536
    assert result["S0"]["transformer_scored_tokens"]==120726
    reused=t2_role_costs(manifest,dict(checkpoints=[dict(c,reused=True) for c in checkpoints]))
    assert reused["all_models"]["lm_head_scored_positions"]==0
