"""Small synthetic files only; no Slurm, model loading, generation or real test labels."""
import builtins
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
import pytest
from cafd import minimal_baseline_report as report


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def write_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))


def cost(jobs,states=None):
    states=states or ["COMPLETED"]*len(jobs)
    rows=[dict(job_id=j,state=state,start_utc=f"2026-09-08T0{i}:00:00",
        end_utc=f"2026-09-08T0{i}:10:00" if state!="RUNNING" else "Unknown",
        elapsed_seconds=600,gpus=1,allocation="gres/gpu:b200=1,gres/gpu=1,mem=192G",
        node="synthetic",exit_code="0:0") for i,(j,state) in enumerate(zip(jobs,states))]
    return dict(job_ids=jobs,rows=rows,gpu_hours=len(jobs)/6 if jobs else None,
        terminal=bool(jobs) and "RUNNING" not in states,
        status="terminal" if "RUNNING" not in states else "accruing",reason=None)


def populate(root,spec,skip_rounds=(),flat_fail=False):
    run=root/"runs/cafd/experiments"/spec["run_id"]
    art=root/"artifacts/cafd/experiments"/spec["run_id"]
    condition=spec["condition"]
    train=[];raw=[];dev=[];devoutputs=[]
    if condition=="s0":
        target=root/"runs/cafd/experiments/mistral_cafd_disjoint_v6/student_base/S0"
        target.mkdir(parents=True,exist_ok=True)
        link=root/"runs/cafd/experiments/mistral_cafd_disjoint_v7/student_base/S0"
        link.parent.mkdir(parents=True,exist_ok=True)
        if not link.exists():link.symlink_to(target,target_is_directory=True)
        cp=target
        selected=0
        write_json(run/"config.json",dict(profile="evaluation_only",condition="s0",total_rounds=0))
        write_json(run/"evaluation_plan.json",dict(checkpoint=str(cp),training_rounds=0,
            actual_optimizer_updates=0,selection=dict(checkpoint=str(cp),round=0,
                selection_rule="predeclared_original_S0",status="frozen")))
    else:
        for step in range(1,201):
            skip=step in skip_rounds or (flat_fail and condition=="grpo")
            train.append(dict(round=step,optimizer_step=not skip))
            for group in range(4):
                for i in range(8):
                    success=step in skip_rounds
                    reward=1.0 if success else (0.0 if flat_fail else (0.1 if i%2 else 0.0))
                    raw.append(dict(round=step,row_id=f"train-{step}-{group}",
                        full_pass=success,reward=reward,completion_ids=[2]))
        for index,step in enumerate(report.MILESTONES):
            correct=min(index,4)
            dev.append(dict(round=step,correct=correct,total=64))
            devoutputs.extend(dict(round=step,row_id=f"dev-{i}",full_pass=i<correct,
                reward=1.0 if i<correct else .1) for i in range(64))
        selected=120
        cp=run/"round120"
        write_rows(art/"training_curve.jsonl",train)
        write_rows(run/"raw_rollouts.jsonl",raw)
        write_rows(art/"development_curve.jsonl",dev)
        write_rows(run/"development_outputs.jsonl",devoutputs)
    outputs=[dict(row_id=f"test-{i}",full_pass=i%2==0,reward=1.0 if i%2==0 else .1,
        family=f"family-{i%3}",completion_ids=[2,3]) for i in range(132)]
    write_rows(art/"final_test_outputs.jsonl",outputs)
    updates=sum(r["optimizer_step"] for r in train)
    identity=dict(training_rounds=spec["rounds"],actual_optimizer_updates=updates,
        selection=dict(selection_rule="predeclared_original_S0" if condition=="s0" else "development_full_pass_earliest_tie",
            round=selected,checkpoint=str(cp),status="frozen"),checkpoint=str(cp))
    write_json(art/"final_scores.json",dict(scores=dict(training_rounds=spec["rounds"],
        actual_optimizer_updates=updates,selected_round=selected,checkpoint=str(cp),correct=66,total=132),
        identity=identity,checkpoint_selection_changed=False))
    job=str(100+next(i for i,s in enumerate(report.SPECS) if s["run_id"]==spec["run_id"]))
    if condition!="s0":write_rows(art/"physical_costs.jsonl",[dict(event="attempt",job_id=job)])
    write_rows(art/"final_test_physical_costs.jsonl",[dict(event="attempt",job_id=job)])
    return run,art


@pytest.fixture
def isolated(monkeypatch,tmp_path):
    monkeypatch.setattr(report,"ROOT",tmp_path)
    monkeypatch.setattr(report,"REPORT",tmp_path/"reports")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(report,"accounting",lambda jobs:cost(jobs))
    return tmp_path


def test_s0_is_legitimate_zero_training_without_complete_or_development(isolated):
    spec=report.SPECS[0]
    run,art=populate(isolated,spec)
    assert not (run/"complete.json").exists()
    result=report.summarize(spec)
    assert result["errors"]==[]
    assert result["training_rounds"]==result["actual_optimizer_updates"]==result["training_answers"]==0
    assert result["development"]==[]
    assert result["test_correct"]==66
    assert "mistral_cafd_disjoint_v6" in result["checkpoint"]


def test_s0_rejects_unplanned_checkpoint_or_development_selection(isolated):
    spec=report.SPECS[0]
    _,art=populate(isolated,spec)
    final=json.loads((art/"final_scores.json").read_text())
    final["scores"]["checkpoint"]=str(isolated/"another_S0")
    final["identity"]["selection"]["selection_rule"]="development_full_pass_earliest_tie"
    write_json(art/"final_scores.json",final)
    assert len(report.summarize(spec)["errors"])==2


@pytest.mark.parametrize("selection",[
    {"rule":"predeclared_original_S0"},
    {"selection_rule":"development_full_pass_earliest_tie","rule":"predeclared_original_S0"},
])
def test_s0_rejects_missing_or_wrong_actual_selection_rule_even_with_legacy_alias(isolated,selection):
    spec=report.SPECS[0]
    _,art=populate(isolated,spec)
    final=json.loads((art/"final_scores.json").read_text())
    final["identity"]["selection"]=dict(selection,round=0,status="frozen",
        checkpoint=final["scores"]["checkpoint"])
    write_json(art/"final_scores.json",final)
    assert report.summarize(spec)["errors"]==["S0 test must use the predeclared original S0 endpoint."]


@pytest.mark.parametrize("condition,expected_kd",[
    ("grpo",0),("absolute_kd",796),("relative_cafd",796),("historical_mpc",0)])
def test_condition_specific_coverage_and_legal_all_success_skip(isolated,condition,expected_kd):
    spec=next(s for s in report.SPECS if s["condition"]==condition)
    populate(isolated,spec,skip_rounds=[194])
    result=report.summarize(spec)
    assert result["errors"]==[]
    assert result["actual_optimizer_updates"]==199 and result["skipped_rounds"]==[194]
    assert result["training_answers"]==6400
    assert result["coverage"]["kd_groups"]==expected_kd
    assert result["coverage"]["rl_groups"]==796


def test_grpo_can_legally_skip_all_flat_fail_rounds(isolated):
    spec=report.SPECS[1]
    populate(isolated,spec,flat_fail=True)
    result=report.summarize(spec)
    assert result["errors"]==[]
    assert result["training_rounds"]==200 and result["actual_optimizer_updates"]==0
    assert result["coverage"]["kd_tokens"]==result["coverage"]["rl_tokens"]==0


def test_illegal_skipped_active_round_is_rejected(isolated):
    spec=report.SPECS[2]
    _,art=populate(isolated,spec)
    train=list(report.records(art/"training_curve.jsonl"))
    train[7]["optimizer_step"]=False
    write_rows(art/"training_curve.jsonl",train)
    final=json.loads((art/"final_scores.json").read_text())
    final["scores"]["actual_optimizer_updates"]=199
    write_json(art/"final_scores.json",final)
    assert any("only legal skip" in e for e in report.summarize(spec)["errors"])


def test_unfrozen_partial_test_does_not_report_partial_accuracy(isolated):
    spec=report.SPECS[0]
    _,art=populate(isolated,spec)
    (art/"final_scores.json").unlink()
    result=report.summarize(spec,partial=True)
    assert result["test_correct"] is None and result["_test"]==[]


def test_attempt_union_deduplicates_shared_training_test_job():
    a=[dict(event="attempt",job_id="1"),dict(event="attempt",job_id="1")]
    b=[dict(event="attempt",job_id="1"),dict(event="attempt",job_id="2")]
    assert report.attempt_ids(a,b)==["1","2"]


@pytest.mark.parametrize("stdout,expected,terminal",[
    ("",None,False),
    ("1|COMPLETED|2026-09-08T00:00:00|2026-09-08T00:00:00|0|gres/gpu=1|node|0:0\n",0,True),
    ("1|RUNNING|2026-09-08T00:00:00|Unknown|3600|gres/gpu=1|node|0:0\n",1,False),
    ("1|COMPLETED|2026-09-08T00:00:00|2026-09-08T01:00:00|3600|gres/gpu=2|node|0:0\n",2,True),
])
def test_sacct_empty_zero_running_and_gpu_multiplicity(monkeypatch,stdout,expected,terminal):
    monkeypatch.setattr(report.subprocess,"run",lambda *a,**k:SimpleNamespace(stdout=stdout))
    result=report.accounting(["1"])
    assert result["gpu_hours"]==expected and result["terminal"]==terminal


def test_sacct_missing_one_job_or_command_failure_remains_unknown(monkeypatch):
    monkeypatch.setattr(report.subprocess,"run",lambda *a,**k:SimpleNamespace(
        stdout="1|COMPLETED|2026-09-08T00:00:00|2026-09-08T01:00:00|3600|gres/gpu=1|n|0:0\n"))
    assert report.accounting(["1","2"])["gpu_hours"] is None
    def fail(*a,**k):raise subprocess.CalledProcessError(1,["sacct"])
    monkeypatch.setattr(report.subprocess,"run",fail)
    assert report.accounting(["1"])["gpu_hours"] is None


def test_shared_condition_job_is_not_double_attributed():
    runs=[dict(run_id="a",accounting=cost(["1"])),dict(run_id="b",accounting=cost(["1"]))]
    shared=report.mark_shared_allocations(runs,[])
    assert shared=={"1":["a","b"]}
    assert all(r["accounting"]["gpu_hours"] is None for r in runs)


def test_smoke_run_ids_sharing_one_engineering_job_are_counted_once(isolated):
    base=isolated/"artifacts/cafd/experiments"
    for run in ["mistral_cafd_minimal_grpo_s2027_smoke","mistral_cafd_minimal_absolute_kd_s2027_smoke"]:
        write_rows(base/run/"physical_costs.jsonl",[dict(event="attempt",job_id="55")])
    result=report.smoke_runs()
    assert len(result)==1 and len(result[0]["included_runs"])==2
    assert result[0]["accounting"]["job_ids"]==["55"]
    assert result[0]["accounting"]["gpu_hours"]==1/6


def test_paired_bootstrap_pairs_by_identity_even_when_order_differs():
    ref=dict(condition="relative_cafd",_test=[dict(row_id=str(i),full_pass=i<20) for i in range(132)])
    new=dict(condition="s0",_test=[dict(row_id=str(i),full_pass=i<30) for i in reversed(range(132))])
    result=report.paired_bootstrap(new,ref)
    assert result["difference"]==pytest.approx(10/132)
    assert result["condition_only"]==10 and result["reference_only"]==0
    assert result==report.paired_bootstrap(new,ref)
    new["_test"][0]["row_id"]="foreign"
    with pytest.raises(ValueError):report.paired_bootstrap(new,ref)


def test_complete_report_requires_five_frozen_rows_and_terminal_refresh(isolated,monkeypatch,capsys):
    for spec in report.SPECS:populate(isolated,spec,skip_rounds=[194] if spec["condition"]=="relative_cafd" else [])
    monkeypatch.setattr(report,"source_proof",lambda runs:{"synthetic":"proof"})
    original_import=builtins.__import__
    def no_models(name,*args,**kwargs):
        if name.split(".")[0] in {"torch","transformers"}:pytest.fail("Model/runtime import attempted")
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",no_models)
    assert report.main(["--final"])==0
    result=json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result["status"]=="complete" and result["frozen_results"]==5
    saved=json.loads((report.REPORT/"comparison.json").read_text())
    assert len(saved["paired_bootstrap"])==3
    text=(report.REPORT/"MINIMAL_BASELINES_FINAL.md").read_text()
    assert "最精简 GRPO 成本" in text and "实际更新次数随合法 skip" in text
    assert "|原始 S0|0|0|0|0|预指定 S0|66/132|" in text
    monkeypatch.setattr(report,"accounting",lambda jobs:cost(jobs,["RUNNING"]*len(jobs)))
    assert report.main(["--final"])==2
    assert report.main(["--partial"])==0
    saved=json.loads((report.REPORT/"comparison.json").read_text())
    assert saved["status"]=="partial" and all(r["accounting"]["gpu_hours"] is not None for r in saved["runs"])
    assert "部分结果" in (report.REPORT/"MINIMAL_BASELINES_FINAL.md").read_text()


def test_missing_result_and_unknown_cost_are_not_zero_or_complete(isolated,monkeypatch):
    populate(isolated,report.SPECS[0])
    monkeypatch.setattr(report,"source_proof",lambda runs:{})
    monkeypatch.setattr(report,"accounting",lambda jobs:dict(job_ids=jobs,rows=[],
        gpu_hours=None,terminal=False,status="unavailable",reason="synthetic missing sacct"))
    assert report.main([])==2
    data=json.loads((report.REPORT/"comparison.json").read_text())
    assert data["status"]=="partial" and sum(r["final_available"] for r in data["runs"])==1
    assert all(r["accounting"]["gpu_hours"] is None for r in data["runs"])
    assert "未知" in (report.REPORT/"MINIMAL_BASELINES_FINAL.md").read_text()


def test_failed_prior_attempt_is_charged_when_latest_recovery_completed():
    r=dict(condition="grpo",run_id="g",new=True,final_available=True,errors=[],
        accounting=cost(["1","2"],["FAILED","COMPLETED"]))
    assert report.ready_for_final([r],[],{})==[]
    assert r["accounting"]["gpu_hours"]==1/3


def test_only_partial_mode_accepts_incomplete_last_journal_line(tmp_path):
    path=tmp_path/"partial.jsonl"
    path.write_text('{"value":1}\n{"partial":')
    assert list(report.records(path,partial=True))==[{"value":1}]
    with pytest.raises(json.JSONDecodeError):list(report.records(path))
