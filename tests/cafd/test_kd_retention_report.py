"""Synthetic report/recovery tests: no real task payloads, models or Slurm calls."""
import builtins
import json
import sys
import types
from pathlib import Path

import pytest

from cafd import kd_retention_report as report
from cafd import kd_retention_finalize as final


def allocation(job, seconds=3600, start="2026-09-08T00:00:00"):
    return dict(job_id=job, elapsed_seconds=seconds, start_utc=start)


def test_training_and_test_attempts_are_counted_once(monkeypatch, tmp_path):
    training=[dict(event="attempt", job_id="10"),dict(event="attempt", job_id="10")]
    testing=[dict(event="attempt", job_id="10"),dict(event="attempt", job_id="11")]
    def fake_lines(path):
        if path.name=="physical_costs.jsonl":return training
        if path.name=="final_test_physical_costs.jsonl":return testing
        return []
    calls=[]
    monkeypatch.setattr(report,"ROOT",tmp_path)
    monkeypatch.setattr(report,"lines",fake_lines)
    def sacct(jobs):
        calls.append(jobs)
        return [allocation("10"),allocation("11",1800)]
    monkeypatch.setattr(report,"accounting",sacct)
    summary=report.summarize(report.NEW)
    assert calls==[["10","11"]]
    assert summary["allocation_totals"]["gpu_hours"]==1.5


@pytest.mark.parametrize("jobs,records,missing,unavailable",[
    ([],[],[],[]),
    (["10"],[],["10"],[]),
    (["10","11"],[allocation("10")],["11"],[]),
    (["10"],[allocation("10",0,"Unknown")],[],["10"]),
])
def test_absent_accounting_is_unknown(jobs,records,missing,unavailable):
    result=report.allocation_totals(jobs,records)
    assert result["gpu_hours"] is None and result["status"]=="unavailable"
    assert result["missing_job_ids"]==missing
    assert result["unavailable_job_ids"]==unavailable


def test_observed_zero_seconds_is_zero_and_duplicate_jobs_not_double_charged():
    assert report.allocation_totals(["10"],[allocation("10",0)])["gpu_hours"]==0
    assert report.allocation_totals(["10","10"],[allocation("10"),allocation("10")])["gpu_hours"]==1


def test_empty_sacct_response_preserves_missing_ids(monkeypatch):
    monkeypatch.setattr(report.subprocess,"run",
        lambda *a,**kw:types.SimpleNamespace(stdout=""))
    rows=report.accounting(["10"])
    assert rows==[]
    assert report.allocation_totals(["10"],rows)["gpu_hours"] is None


def test_matched_compute_requires_both_complete_costs():
    def run(name,hours):
        return dict(run=name,allocation_totals={"gpu_hours":hours},
            development=[dict(checkpoint_available=True,cumulative_gpu_hours=.2,
                correct=3,round=10)])
    missing=report.matched_compute([run("old",1),run("new",None)])
    assert missing["common_total_gpu_hours"] is None
    assert missing["status"]=="unavailable"
    assert all(r["best_observed_development_with_saved_checkpoint"] is None for r in missing["rows"])
    known=report.matched_compute([run("old",1),run("new",2)])
    assert known["common_total_gpu_hours"]==1
    assert all(r["best_observed_development_with_saved_checkpoint"]["round"]==10 for r in known["rows"])


@pytest.mark.parametrize("formal_hours",[None,0.0])
@pytest.mark.parametrize("new_updates",[200,199])
def test_final_markdown_handles_unknown_and_zero_costs_without_models(monkeypatch,tmp_path,capsys,formal_hours,new_updates):
    coverage={k:0 for k in ["groups","tokens","actual_kd_groups","actual_kd_tokens",
        "actual_kd_group_fraction","actual_kd_token_fraction","all_fail_variable_groups",
        "all_fail_variable_tokens"]}
    def summary(run):
        updates=new_updates if run==report.NEW else 200
        complete=dict(rounds=200,actual_updates=updates)
        if updates==199:
            complete["protocol_clarification"]=approved_clarification()
        return dict(run=run,completed_rounds=200,actual_updates=updates,raw_count=6400,
            complete=complete,groups=[{}]*800,coverage=coverage,coverage_by_block=[],
            allocation_totals={"gpu_hours":formal_hours},
            development=[dict(run=run,round=step,correct=1,total=64,mean_reward=.2,
                cumulative_gpu_hours=0,checkpoint_available=True,saved_per_task_count=64)
                for step in report.MILESTONES],
            final=dict(scores=dict(selected_round=0,correct=66,checkpoint="/synthetic/S0"),
                by_family=[dict(family="synthetic",correct=66,total=132)],
                failure_counts={"passed":66,"semantic":66}))
    rows=[dict(row_id=f"synthetic-{i}",full_pass=(i%2==0)) for i in range(132)]
    def saved_lines(path):
        if path.name=="final_test_outputs.jsonl":return rows
        if path.name=="physical_costs.jsonl":return [dict(event="attempt",job_id="99")]
        return []
    monkeypatch.setattr(report,"ROOT",tmp_path)
    monkeypatch.setattr(report,"REPORT",tmp_path/"reports")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(report,"summarize",summary)
    monkeypatch.setattr(report,"lines",saved_lines)
    monkeypatch.setattr(report,"read",lambda path:{"status":"SMOKE_PASSED"})
    monkeypatch.setattr(report,"accounting",lambda jobs:[])
    monkeypatch.setattr(report,"program_stats",lambda records,tok:{"synthetic":True})
    monkeypatch.setitem(sys.modules,"cafd.mistral_runtime",
        types.SimpleNamespace(load_tokenizers=lambda path:(None,None)))
    monkeypatch.setattr(sys,"argv",["kd_retention_report"])
    report.main()
    destination=tmp_path/"reports"
    text=(destination/"CAFD_KD_RETENTION_FINAL.md").read_text()
    returned=json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert returned["rounds"]==200 and returned["updates"]==new_updates
    assert f"Completed 200 rounds and {new_updates} optimizer updates" in text
    assert "protocol_clarification_20260908/report_pytest.xml" in text
    if new_updates==199:
        assert returned["protocol_clarification"]==approved_clarification()
        assert "Historical baseline completed 200 rounds, 200 optimizer updates" in text
        assert "this run completed 200 rounds, 199 optimizer updates" in text
        assert "optimizer-update counts differ" in text
        assert "all 32 answers in round194 fully passed" in text
        assert "No extra training was authorized or added" in text
        assert "Both runs completed 200 rounds and 200 optimizer updates" not in text
    else:
        assert returned["protocol_clarification"] is None
    assert "Engineering smoke allocation: missing GPU h" in text
    matched=json.loads((destination/"matched_compute.json").read_text())
    if formal_hours is None:
        assert "Allocated Student run costs: old missing GPU h; new missing GPU h" in text
        assert "no matched-compute conclusion" in text
        assert matched["common_total_gpu_hours"] is None
    else:
        assert "Allocated Student run costs: old 0.000000 GPU h; new 0.000000 GPU h" in text
        assert matched["common_total_gpu_hours"]==0
    assert json.loads((destination/"engineering_cost.json").read_text())["smoke_allocation_totals"]["gpu_hours"] is None


def test_completed_journal_recovery_marks_complete_without_model_or_test_reload(monkeypatch,tmp_path):
    artifact=tmp_path/"artifact"
    artifact.mkdir()
    identity=dict(test_ids=[f"synthetic-{i}" for i in range(132)],
        checkpoint=str(tmp_path/"checkpoint"),run_id=report.NEW,
        actual_optimizer_updates=200,selection={"round":40})
    context=dict(artifact=str(artifact),identity=identity,training_costs={},config={},limitations=[])
    records=[dict(index=i,row_id=identity["test_ids"][i],checkpoint=identity["checkpoint"],
        run_id=identity["run_id"],family=final.FAMILIES[i%3],reward=1.0,
        full_pass=True,prompt_ids=[1],completion_ids=[2],failure_category="passed") for i in range(132)]
    journal=artifact/"final_test_outputs.jsonl"
    journal.write_text("".join(json.dumps(row)+"\n" for row in records))
    progress=artifact/"final_test_progress.json"
    progress.write_text(json.dumps({"status":"evaluating","completed":132,"total":132}))
    monkeypatch.setattr(final,"ROOT",tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(final,"validate_finalization",lambda *args:context)
    monkeypatch.setattr(final,"load_test_rows",lambda *args:pytest.fail("test payload reloaded"))
    monkeypatch.setattr(final,"evaluate_remaining",lambda *args:pytest.fail("generation attempted"))
    original_import=builtins.__import__
    def no_models(name,*args,**kwargs):
        if name=="torch" or name in {"mpc_runtime","mistral_runtime"}:
            pytest.fail(f"model import attempted: {name}")
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",no_models)
    monkeypatch.setattr(sys,"argv",["kd_retention_finalize","--run-id",report.NEW])
    journal_before=journal.read_bytes()
    final.main()
    status=json.loads(progress.read_text())
    assert status==dict(status="complete",completed=132,total=132,correct=132,
        checkpoint=identity["checkpoint"])
    paths=["final_test_summary.json","final_scores.json","final_scores.csv",
        "final_scores_by_family.csv","final_test_costs.json"]
    sealed={name:(artifact/name).read_bytes() for name in paths}
    final.main()
    assert journal.read_bytes()==journal_before
    assert {name:(artifact/name).read_bytes() for name in paths}==sealed
    assert json.loads(progress.read_text())==status


def approved_clarification():
    return {
        "authorization":"user_approved_200_rounds_with_199_actual_updates",
        "preserved_round_budget":200,
        "actual_optimizer_updates":199,
        "skipped_all_success_rounds":[194],
        "additional_training_authorized":False,
    }


def approved_summary():
    return dict(completed_rounds=200,actual_updates=199,
        complete=dict(rounds=200,actual_updates=199,
            protocol_clarification=approved_clarification()))


def test_200_updates_needs_no_protocol_exception():
    assert report.validate_report_training_budget(dict(completed_rounds=200,actual_updates=200)) is None


def test_exact_user_authorization_admits_199_updates():
    assert report.validate_report_training_budget(approved_summary())==approved_clarification()


@pytest.mark.parametrize("mutation",[
    None,
    {"authorization":"inferred_by_agent"},
    {"preserved_round_budget":201},
    {"actual_optimizer_updates":198},
    {"skipped_all_success_rounds":[193]},
    {"additional_training_authorized":True},
    {"unexpected_field":"not_an_exact_match"},
    {"preserved_round_budget":200.0},
    {"actual_optimizer_updates":199.0},
    {"skipped_all_success_rounds":[194.0]},
    {"additional_training_authorized":0},
])
def test_199_updates_rejects_missing_or_nonmatching_authorization(mutation):
    summary=approved_summary()
    if mutation is None:
        summary["complete"].pop("protocol_clarification")
    else:
        summary["complete"]["protocol_clarification"].update(mutation)
    with pytest.raises(RuntimeError,match="exact user-approved"):
        report.validate_report_training_budget(summary)


@pytest.mark.parametrize("updates",[198,201,199.0,200.0,True])
def test_exception_does_not_admit_other_update_counts_or_types(updates):
    summary=approved_summary()
    summary["actual_updates"]=updates
    with pytest.raises(RuntimeError):
        report.validate_report_training_budget(summary)


@pytest.mark.parametrize("target,key,value",[
    ("summary","completed_rounds",199),
    ("summary","completed_rounds",200.0),
    ("complete","rounds",199),
    ("complete","actual_updates",200),
    ("complete","actual_updates",199.0),
])
def test_authorization_does_not_override_conflicting_completion_metadata(target,key,value):
    summary=approved_summary()
    (summary if target=="summary" else summary["complete"])[key]=value
    with pytest.raises(RuntimeError):
        report.validate_report_training_budget(summary)
