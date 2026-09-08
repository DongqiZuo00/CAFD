"""Read-only output analysis: no model load, generation, training or test selection."""
import argparse,csv,datetime,hashlib,json,os,subprocess
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

ROOT=Path("/blue/du.j/jinjiaguo/CAFD")
OLD="mistral_cafd_mpc_v1_formal"
NEW="mistral_cafd_kd_retention_replay_s2027"
MILESTONES=[0,10,40,80,120,160,200]
REPORT=ROOT/"reports/kd_retention_replay_s2027"

def read(path):
    return json.loads(path.read_text())
def lines(path):
    if not path.exists():return []
    with path.open() as f:return [json.loads(l) for l in f if l.strip()]
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n")
def table(path,rows):
    if not rows:return
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def accounting(jobs):
    if not jobs:return []
    result=subprocess.run(["sacct","-X","-n","-P","-j",",".join(jobs),
        "-o","JobIDRaw,State,Start,End,ElapsedRaw,AllocTRES,NodeList,ExitCode"],
        env=dict(os.environ,TZ="UTC"),capture_output=True,text=True,check=True)
    output=[]
    for line in result.stdout.splitlines():
        parts=line.split("|")
        if len(parts)<8 or parts[0] not in jobs:continue
        jid,state,start,end,elapsed,tres,node,exitcode=parts[:8]
        output.append(dict(job_id=jid,state=state,start_utc=start,end_utc=end,
            elapsed_seconds=int(elapsed),allocation=tres,node=node,exit_code=exitcode))
    return output

def attempt_job_ids(*ledgers):
    """Training and test may occupy separate recovery allocations."""
    return list(dict.fromkeys(str(r["job_id"]) for ledger in ledgers for r in ledger
        if r.get("event")=="attempt" and r.get("job_id")))

def allocation_totals(jobs,allocations):
    """Missing accounting is unknown; an observed elapsed=0 remains a real zero."""
    jobs=list(dict.fromkeys(jobs))
    by_job={a["job_id"]:a for a in allocations if a["job_id"] in jobs}
    missing=[job for job in jobs if job not in by_job]
    unavailable=[job for job,a in by_job.items()
        if a.get("start_utc") in (None,"Unknown","None")
        or type(a.get("elapsed_seconds")) is not int or a["elapsed_seconds"]<0]
    known=bool(jobs) and not missing and not unavailable
    return dict(job_ids=jobs,missing_job_ids=missing,unavailable_job_ids=unavailable,
        gpu_hours=sum(by_job[job]["elapsed_seconds"] for job in jobs)/3600 if known else None,
        status="available" if known else "unavailable")

def matched_compute(runs):
    costs=[r["allocation_totals"]["gpu_hours"] for r in runs]
    available=all(cost is not None for cost in costs)
    common=min(costs) if available else None
    rows=[]
    for r in runs:
        candidates=[] if common is None else [
            d for d in r["development"] if d["checkpoint_available"]
            and d["cumulative_gpu_hours"] is not None and d["cumulative_gpu_hours"]<=common]
        best=max(candidates,key=lambda d:(d["correct"],-d["round"])) if candidates else None
        rows.append(dict(run=r["run"],best_observed_development_with_saved_checkpoint=best))
    return dict(common_total_gpu_hours=common,rows=rows,
        status="available" if available else "unavailable",
        reason=None if available else "Missing Slurm accounting; no matched-compute conclusion.",
        basis="Total allocated Student training+control+development+test reservation; no interpolation or new evaluation; compares only observed development results at retained checkpoints.",
        excludes="Existing Teacher trajectory training and separately reported engineering smoke.")

def cumulative_time(event,allocations):
    seconds=0.
    for a in allocations:
        if a["start_utc"] in ("Unknown","None"):continue
        start=datetime.datetime.fromisoformat(a["start_utc"]).replace(tzinfo=datetime.timezone.utc).timestamp()
        seconds+=max(0.,min(event-start,a["elapsed_seconds"]))
    return seconds/3600.

def coverage(raw):
    groups=defaultdict(list)
    for r in raw:groups[(r["round"],r["row_id"])].append(r)
    total=Counter(); group_rows=[];by_block=defaultdict(Counter)
    for (step,identifier),g in groups.items():
        rewards=[r["reward"] for r in g];flags=[r["full_pass"] for r in g]
        varied=max(rewards)>min(rewards);fail=not any(flags)
        old_kd=fail and not varied
        n=sum(len(r["completion_ids"]) for r in g)
        row=dict(round=step,row_id=identifier,family=g[0]["family"],tokens=n,
            rewards=rewards,full_pass=flags,all_fail_variable=fail and varied,
            old_route="KD" if old_kd else ("RL" if varied else "skip"),
            retention_route="KD+RL" if fail and varied else ("KD" if fail else ("RL" if varied else "skip")))
        group_rows.append(row)
        c=Counter(groups=1,tokens=n,all_fail_variable_groups=int(fail and varied),
            all_fail_variable_tokens=n*int(fail and varied),
            old_kd_groups=int(old_kd),old_kd_tokens=n*int(old_kd),
            retention_kd_groups=int(fail),retention_kd_tokens=n*int(fail),
            rl_groups=int(varied),rl_tokens=n*int(varied),skip_groups=int(not fail and not varied))
        total.update(c);by_block[(step-1)//10+1].update(c)
    return dict(total),group_rows,[dict(block=b,**v) for b,v in sorted(by_block.items())]

def summarize(run):
    out=ROOT/"runs/cafd/experiments"/run
    art=ROOT/"artifacts/cafd/experiments"/run
    raw=lines(out/"raw_rollouts.jsonl")
    train=lines(art/"training_curve.jsonl")
    dev=lines(art/"development_curve.jsonl")
    outputs=lines(out/"development_outputs.jsonl")
    ledger=lines(art/"physical_costs.jsonl")
    test_ledger=lines(art/"final_test_physical_costs.jsonl")
    jobs=attempt_job_ids(ledger,test_ledger)
    allocations=accounting(jobs)
    allocation_cost=allocation_totals(jobs,allocations)
    events=[r for r in ledger if r.get("name")=="development_generated_tokens"]
    by_step=defaultdict(list)
    for r in outputs:by_step[r["round"]].append(r)
    devrows=[]
    for index,r in enumerate(dev):
        records=by_step[r["round"]]
        # Old, uninterrupted formal baseline: 64 physical evaluation events per milestone.
        if r.get("cumulative_allocated_gpu_hours") is not None:
            h=r["cumulative_allocated_gpu_hours"]
        elif run==OLD and len(events)==448 and allocation_cost["gpu_hours"] is not None:
            h=cumulative_time(events[(index+1)*64-1]["time"],allocations)
        else:h=None
        path=ROOT/"runs/cafd/experiments/mistral_cafd_disjoint_v7/student_base/S0" if r["round"]==0 else out/f"round{r['round']}"
        devrows.append(dict(run=run,round=r["round"],correct=r["correct"],total=r["total"],
            mean_reward=(sum(x["reward"] for x in records)/len(records)) if len(records)==64 else r.get("mean_reward"),
            cumulative_gpu_hours=h,checkpoint_available=path.is_dir(),
            saved_per_task_count=len(records)))
    cov,group_rows,blocks=coverage(raw)
    new=run==NEW
    cov.update(actual_kd_groups=cov.get("retention_kd_groups" if new else "old_kd_groups",0),
        actual_kd_tokens=cov.get("retention_kd_tokens" if new else "old_kd_tokens",0))
    if cov.get("groups"):
        cov["actual_kd_group_fraction"]=cov["actual_kd_groups"]/cov["groups"]
        cov["actual_kd_token_fraction"]=cov["actual_kd_tokens"]/cov["tokens"]
    final=read(art/"final_scores.json") if (art/"final_scores.json").exists() else None
    return dict(run=run,completed_rounds=len(train),actual_updates=sum(r["optimizer_step"] for r in train),
        development=devrows,coverage=cov,coverage_by_block=blocks,groups=group_rows,
        allocation=allocations,allocation_totals=allocation_cost,complete=read(out/"complete.json") if (out/"complete.json").exists() else None,
        final=final,raw_count=len(raw))

def validate_report_training_budget(summary):
    """Admit the one explicitly approved all-success skip; never invent updates."""
    if type(summary["completed_rounds"]) is not int or summary["completed_rounds"]!=200:
        raise RuntimeError("final report requires 200 completed rounds")
    updates=summary["actual_updates"]
    if type(updates) is int and updates==200:
        return None
    expected={
        "authorization":"user_approved_200_rounds_with_199_actual_updates",
        "preserved_round_budget":200,
        "actual_optimizer_updates":199,
        "skipped_all_success_rounds":[194],
        "additional_training_authorized":False,
    }
    complete=summary.get("complete")
    clarification=complete.get("protocol_clarification") if isinstance(complete,dict) else None
    if (type(updates) is not int or updates!=199 or type(clarification) is not dict
            or clarification!=expected
            or any(type(clarification.get(k)) is not type(v) for k,v in expected.items())
            or any(type(step) is not int for step in clarification["skipped_all_success_rounds"])
            or type(complete.get("rounds")) is not int or complete["rounds"]!=200
            or type(complete.get("actual_updates")) is not int or complete["actual_updates"]!=199):
        raise RuntimeError("199 updates require the exact user-approved protocol clarification")
    return clarification

def program_stats(records,tokenizer):
    from .verifier import extract_contract_program
    counts=Counter();raw_counts=Counter();family=defaultdict(Counter)
    for r in records:
        text=tokenizer.decode(r["completion_ids"],skip_special_tokens=True)
        program=extract_contract_program(text)
        if program is None:continue
        counts[program]+=1;family[r["family"]][program]+=1
        raw_counts[tuple(r["completion_ids"])]+=1
    n=sum(counts.values())
    return dict(valid_contract_outputs=n,unique_programs=len(counts),
        duplicates_beyond_first=n-len(counts),
        duplicate_fraction=(n-len(counts))/n if n else None,
        largest_identical_program_count=max(counts.values(),default=0),
        top_programs=[dict(program=p,count=n,sha256=hashlib.sha256(p.encode()).hexdigest()) for p,n in counts.most_common(5)],
        by_family={f:dict(outputs=sum(c.values()),unique_programs=len(c),largest_count=max(c.values(),default=0)) for f,c in family.items()},
        definition="Exact equality of official extract_contract_program text; surrounding whitespace stripped; no semantic canonicalization.",
        old_definition_available=False,
        limitation="No explicit historical duplicate-program definition was found in the formal report/source archive; this identical post-hoc rule is applied to both raw output sets.")

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--partial",action="store_true");args=parser.parse_args()
    if Path.cwd().resolve()!=ROOT:raise RuntimeError("CAFD only")
    REPORT.mkdir(exist_ok=True)
    old=summarize(OLD);new=summarize(NEW)
    write(REPORT/"comparison_progress.json",dict(old=old,new=new))
    table(REPORT/"development_comparison.csv",old["development"]+new["development"])
    table(REPORT/"coverage_by_block.csv",[dict(run=r["run"],**b) for r in [old,new] for b in r["coverage_by_block"]])
    if args.partial and not new["final"]:
        print(json.dumps(dict(rounds=new["completed_rounds"],actual_updates=new["actual_updates"],status="partial")));return
    assert new["final"] is not None
    protocol_clarification=validate_report_training_budget(new)
    assert new["raw_count"]==6400 and len(new["groups"])==800
    oldtest=lines(ROOT/"artifacts/cafd/experiments"/OLD/"final_test_outputs.jsonl")
    newtest=lines(ROOT/"artifacts/cafd/experiments"/NEW/"final_test_outputs.jsonl")
    assert len(oldtest)==len(newtest)==132
    assert [r["row_id"] for r in oldtest]==[r["row_id"] for r in newtest]
    delta=np.asarray([int(b["full_pass"])-int(a["full_pass"]) for a,b in zip(oldtest,newtest)])
    rng=np.random.default_rng(2027)
    boot=delta[rng.integers(0,132,size=(20000,132))].mean(axis=1)
    ci=np.quantile(boot,[.025,.975]).tolist()
    paired=dict(new_minus_old=float(delta.mean()),bootstrap_95=ci,replicates=20000,
        seed=2027,paired_by="row_id",old_only=int(sum(d<0 for d in delta)),new_only=int(sum(d>0 for d in delta)),
        both_pass=sum(a["full_pass"] and b["full_pass"] for a,b in zip(oldtest,newtest)),
        limitation="One training seed; item bootstrap does not capture training randomness.")
    write(REPORT/"paired_test_bootstrap.json",paired)
    from .mistral_runtime import load_tokenizers
    _,tok=load_tokenizers(ROOT/".cache/huggingface/hub")
    repeats={}
    for run,test in [(OLD,oldtest),(NEW,newtest)]:
        repeats[run]={"test":program_stats(test,tok)}
        out=ROOT/"runs/cafd/experiments"/run
        repeats[run]["training"]=program_stats(lines(out/"raw_rollouts.jsonl"),tok)
        development=lines(out/"development_outputs.jsonl")
        repeats[run]["development"]={str(step):program_stats([r for r in development if r["round"]==step],tok)
            for step in MILESTONES if any(r["round"]==step for r in development)}
    write(REPORT/"repeated_programs.json",repeats)
    costs=[r["allocation_totals"]["gpu_hours"] for r in [old,new]]
    matched=matched_compute([old,new])
    common=matched["common_total_gpu_hours"]
    write(REPORT/"matched_compute.json",matched)
    smoke=read(ROOT/"runs/cafd/experiments"/(NEW+"_smoke")/"complete.json")
    smokejobs=attempt_job_ids(lines(ROOT/"artifacts/cafd/experiments"/(NEW+"_smoke")/"physical_costs.jsonl"))
    smokealloc=accounting(smokejobs)
    smoketotal=allocation_totals(smokejobs,smokealloc)
    write(REPORT/"engineering_cost.json",dict(smoke_allocations=smokealloc,smoke_allocation_totals=smoketotal,smoke_result=smoke,
        cpu_test_file=str(REPORT/"cpu_tests.xml"),
        cpu_test_files=[str(REPORT/"cpu_tests.xml"),
            str(REPORT/"report_engineering_20260908/pytest.xml"),
            str(REPORT/"protocol_clarification_20260908/report_pytest.xml")],
        cpu_note="CPU pytest elapsed is recorded in JUnit; engineering analysis CPU wall time is not comprehensively instrumented."))
    oldscore=old["final"]["scores"];newscore=new["final"]["scores"]
    devold={r["round"]:r for r in old["development"]};devnew={r["round"]:r for r in new["development"]}
    def fmt(v):return "missing" if v is None else f"{v:.6f}"
    budget_note=(
        f"Historical baseline completed {old['completed_rounds']} rounds, {old['actual_updates']} optimizer updates and {old['raw_count']} Student answers; this run completed {new['completed_rounds']} rounds, {new['actual_updates']} optimizer updates and {new['raw_count']} Student answers. The comparison shares the 200-round and 6400-answer budgets; optimizer-update counts differ."
        if protocol_clarification is not None else
        f"Both runs completed {new['completed_rounds']} rounds and {new['actual_updates']} optimizer updates; each generated 6400 Student training answers.")
    authorization_note=(
        "User-approved protocol clarification: all 32 answers in round194 fully passed, so every group used the unchanged all-success skip rule. The user approved retaining 200 rounds with 199 actual optimizer updates and proceeding to the single frozen test. No extra training was authorized or added."
        if protocol_clarification is not None else
        "The original 200-round, 200-update training budget was completed.")
    text=[
        "# KD coverage modification + old schedule replay",
        "",
        f"Completed {new['completed_rounds']} rounds and {new['actual_updates']} optimizer updates; 800 prompt groups and 6400 Student answers.",
        budget_note,
        authorization_note,
        f"Development selected round{newscore['selected_round']}; one frozen test: {newscore['correct']}/132. Historical baseline selected round{oldscore['selected_round']}, test {oldscore['correct']}/132.",
        "This run used the old realized teacher schedule, not adaptive MPC. Control observations and MPC recommendations remain diagnostic.",
        "",
        "| round | old correct/64 | new correct/64 | old mean reward | new mean reward | old cumulative GPU h | new cumulative GPU h |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for step in MILESTONES:
        a,b=devold[step],devnew[step]
        text.append(f"|{step}|{a['correct']}/64|{b['correct']}/64|{fmt(a['mean_reward'])}|{fmt(b['mean_reward'])}|{fmt(a['cumulative_gpu_hours'])}|{fmt(b['cumulative_gpu_hours'])}|")
    text+=["","Old round0 per-task development outputs were not saved; its mean reward is unavailable.",
        "Old milestone allocated times reconstructed from the last of each 64 development-generation events and UTC Slurm allocation start. New milestone times are recorded directly.",
        "","| test family | old correct/total | new correct/total |","|---|---:|---:|"]
    for a,b in zip(old["final"]["by_family"],new["final"]["by_family"]):
        assert a["family"]==b["family"]
        text.append(f"|{a['family']}|{a['correct']}/{a['total']}|{b['correct']}/{b['total']}|")
    text+=["",f"Test difference (new-old): {paired['new_minus_old']*100:.3f} percentage points; item-paired bootstrap 95% interval [{ci[0]*100:.3f}, {ci[1]*100:.3f}] points (20,000 resamples). This interval excludes training-seed uncertainty.",
        "","| coverage | old | new |","|---|---:|---:|"]
    for k in ["groups","tokens","actual_kd_groups","actual_kd_tokens","actual_kd_group_fraction","actual_kd_token_fraction","all_fail_variable_groups","all_fail_variable_tokens"]:
        text.append(f"|{k}|{old['coverage'][k]}|{new['coverage'][k]}|")
    text+=["","Routing counterfactuals use each run's own outputs; they do not substitute for the old policy's training results.",
        "",f"Allocated Student run costs: old {fmt(costs[0])} GPU h; new {fmt(costs[1])} GPU h. Existing Teacher trajectory cost excluded.",
        f"Engineering smoke allocation: {fmt(smoketotal['gpu_hours'])} GPU h; separate from the formal run.",
        (f"Common observed total budget: {fmt(common)} GPU h. See matched_compute.json for only measured, retained checkpoints; no interpolated performance."
         if common is not None else "Common GPU budget unavailable because Slurm accounting is incomplete; no matched-compute conclusion is reported."),
        "New costs.json separately reports rollout, frozen scoring, teacher loading, optimization, control, development, probe setup and resume saving. Wall-clock operation bins are not exclusive hardware kernel timings: target vocabulary projections/recomputed backward remain in optimization; initialization_inclusive_seconds overlaps startup operations. GPU reservation totals are authoritative.",
        "","Test failure counts: old "+json.dumps(old["final"]["failure_counts"])+"; new "+json.dumps(new["final"]["failure_counts"])+".",
        "Repeated-program statistics use exact extracted program equality for both sets; historical formal artifacts did not define a repetition metric. See repeated_programs.json for top programs and per-family counts.",
        "","Single-seed historical comparison, not a multi-seed causal validation. Test tasks have historical evaluations; all 64 development tasks were exposed to Teacher training (44 SFT, 20 RL).",
        "Any improvement first supports the supervision-coverage hypothesis; relative-versus-absolute target superiority remains untested. No additional variants or expanded budget were run.",
        "","## Locations",
        f"- Code: {ROOT}/cafd/kd_retention_*.py",
        f"- Minimal differences and validation: {REPORT}",
        f"- Initial CPU validation: {REPORT}/cpu_tests.xml",
        f"- Report cost/recovery validation: {REPORT}/report_engineering_20260908/pytest.xml",
        f"- Authorized round/update boundary validation: {REPORT}/protocol_clarification_20260908/report_pytest.xml",
        f"- Authorized report-only changes: {REPORT}/protocol_clarification_20260908/report.patch",
        f"- Resume: {ROOT}/runs/cafd/experiments/{NEW}/resume.pt",
        f"- Selected model: {newscore['checkpoint']}",
        f"- Training groups/control/development outputs: {ROOT}/runs/cafd/experiments/{NEW}",
        f"- Training curve/costs/test outputs: {ROOT}/artifacts/cafd/experiments/{NEW}",
        "",
        "Known CSV field-order finalization issue fixed in the isolated finalization module. Existing 132 saved baseline outputs re-sealed twice in a scratch fixture without changes; formal finalization is invoked twice and must not regenerate completed test outputs."
    ]
    (REPORT/"CAFD_KD_RETENTION_FINAL.md").write_text("\n".join(text)+"\n")
    print(json.dumps(dict(rounds=new["completed_rounds"],updates=new["actual_updates"],protocol_clarification=protocol_clarification,selected_round=newscore["selected_round"],test_correct=newscore["correct"],report=str(REPORT/"CAFD_KD_RETENTION_FINAL.md"))))
if __name__=="__main__":main()
