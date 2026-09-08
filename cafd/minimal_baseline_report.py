"""Saved-artifact comparison only: no model loading, training or generation."""
from __future__ import annotations
import argparse, csv, datetime, hashlib, json, os, re, subprocess
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path("/blue/du.j/jinjiaguo/CAFD")
REPORT=ROOT/"reports/minimal_baselines_s2027"
MILESTONES=[0,10,40,80,120,160,200]
SPECS=[
    dict(condition="s0",label="原始 S0",run_id="mistral_cafd_minimal_s0_s2027",rounds=0,new=True),
    dict(condition="grpo",label="纯 GRPO（保留诊断）",run_id="mistral_cafd_minimal_grpo_s2027",rounds=200,new=True),
    dict(condition="absolute_kd",label="绝对目标 KD+RL",run_id="mistral_cafd_minimal_absolute_kd_s2027",rounds=200,new=True),
    dict(condition="relative_cafd",label="相对目标 CAFD",run_id="mistral_cafd_kd_retention_replay_s2027",rounds=200,new=False),
    dict(condition="historical_mpc",label="历史 CAFD-MPC",run_id="mistral_cafd_mpc_v1_formal",rounds=200,new=False),
]
ACTIVE_STATES={"PENDING","RUNNING","COMPLETING","CONFIGURING","SUSPENDED","RESIZING","REQUEUED"}

def read(path):
    return json.loads(path.read_text()) if path.exists() else None

def records(path,partial=False):
    if not path.exists():return
    with path.open() as handle:
        while True:
            line=handle.readline()
            if not line:break
            if not line.strip():continue
            try:yield json.loads(line)
            except json.JSONDecodeError:
                if partial and not handle.read().strip():return
                raise

def attempt_ids(*ledgers):
    return list(dict.fromkeys(str(r["job_id"]) for ledger in ledgers for r in ledger
        if r.get("event")=="attempt" and r.get("job_id")))

def accounting(jobs):
    jobs=list(dict.fromkeys(jobs))
    if not jobs:return dict(job_ids=[],rows=[],gpu_hours=None,terminal=False,
        status="unavailable",reason="No recorded allocation.")
    try:
        result=subprocess.run(["sacct","-X","-n","-P","-j",",".join(jobs),"-o",
            "JobIDRaw,State,Start,End,ElapsedRaw,AllocTRES,NodeList,ExitCode"],
            env=dict(os.environ,TZ="UTC"),capture_output=True,text=True,check=True)
    except (OSError,subprocess.CalledProcessError) as exc:
        return dict(job_ids=jobs,rows=[],gpu_hours=None,terminal=False,status="unavailable",
            reason=f"Slurm accounting unavailable: {type(exc).__name__}")
    by_job={}
    for line in result.stdout.splitlines():
        a=line.split("|")
        if len(a)<8 or a[0] not in jobs:continue
        gpu=re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)",a[5])
        if gpu is None:gpu=re.search(r"(?:^|,)gres/gpu:b200=(\d+)(?:,|$)",a[5])
        seconds=int(a[4]) if a[4].isdigit() else None
        count=int(gpu.group(1)) if gpu else None
        by_job[a[0]]=dict(job_id=a[0],state=a[1],start_utc=a[2],end_utc=a[3],
            elapsed_seconds=seconds,gpus=count,allocation=a[5],node=a[6],exit_code=a[7])
    rows=[by_job[j] for j in jobs if j in by_job]
    known=len(rows)==len(jobs) and all(a["start_utc"] not in ("Unknown","None","")
        and a["elapsed_seconds"] is not None and a["gpus"] is not None for a in rows)
    terminal=known and all(a["state"].split()[0] not in ACTIVE_STATES
        and a["end_utc"] not in ("Unknown","None","") for a in rows)
    return dict(job_ids=jobs,rows=rows,
        gpu_hours=sum(a["elapsed_seconds"]*a["gpus"] for a in rows)/3600 if known else None,
        terminal=terminal,status="terminal" if terminal else ("accruing" if known else "unavailable"),
        reason=None if known else "Missing/incomplete Slurm rows; unknown costs are not zero.")

def routing(condition,rewards,flags):
    varied=max(rewards)>min(rewards)
    all_fail=not any(flags)
    kd=(all_fail and not varied) if condition=="historical_mpc" else (
        all_fail if condition in {"relative_cafd","absolute_kd"} else False)
    return kd,varied,all_fail and varied

def summarize(spec,partial=False):
    run=ROOT/"runs/cafd/experiments"/spec["run_id"]
    art=ROOT/"artifacts/cafd/experiments"/spec["run_id"]
    train=list(records(art/"training_curve.jsonl",partial))
    ledger=list(records(art/"physical_costs.jsonl",partial))
    test_ledger=list(records(art/"final_test_physical_costs.jsonl",partial))
    groups=defaultdict(lambda:dict(rewards=[],flags=[],tokens=0))
    for r in records(run/"raw_rollouts.jsonl",partial):
        g=groups[(r["round"],r["row_id"])]
        g["rewards"].append(r["reward"]);g["flags"].append(r["full_pass"])
        g["tokens"]+=len(r["completion_ids"])
    coverage=Counter(groups=0,answers=0,tokens=0,kd_groups=0,kd_tokens=0,rl_groups=0,
        rl_tokens=0,overlap_groups=0,overlap_tokens=0,all_fail_variable_groups=0)
    active=defaultdict(bool)
    for (step,_),g in groups.items():
        kd,rl,overlap_eligible=routing(spec["condition"],g["rewards"],g["flags"])
        n=g["tokens"]
        coverage.update(groups=1,answers=len(g["flags"]),tokens=n,kd_groups=int(kd),kd_tokens=n*kd,
            rl_groups=int(rl),rl_tokens=n*rl,overlap_groups=int(kd and rl),
            overlap_tokens=n*(kd and rl),all_fail_variable_groups=int(overlap_eligible))
        active[step]|=(kd or rl) and n>0
    updates=sum(bool(r["optimizer_step"]) for r in train)
    physical=Counter()
    for r in ledger:
        if r.get("event")=="resource":physical[r["name"]]+=r["delta"]
    dev_outputs=defaultdict(list)
    for r in records(run/"development_outputs.jsonl",partial):dev_outputs[r["round"]].append(r)
    dev=[]
    for r in records(art/"development_curve.jsonl",partial):
        saved=dev_outputs[r["round"]]
        dev.append(dict(condition=spec["condition"],run_id=spec["run_id"],round=r["round"],
            correct=r["correct"],total=r["total"],
            mean_reward=sum(x["reward"] for x in saved)/len(saved) if saved else r.get("mean_reward"),
            saved_outputs=len(saved),unique_ids=len({x["row_id"] for x in saved}),
            cumulative_allocated_gpu_hours=r.get("cumulative_allocated_gpu_hours")))
    final=read(art/"final_scores.json")
    test=list(records(art/"final_test_outputs.jsonl")) if final else []
    errors=[]
    if final:
        score=final["scores"]
        if len(test)!=132 or len({r["row_id"] for r in test})!=132:errors.append("Frozen test must contain 132 unique saved task outputs.")
        if score.get("correct")!=sum(r["full_pass"] for r in test) or score.get("total")!=132:errors.append("Frozen test score differs from saved outputs.")
        if score.get("training_rounds")!=spec["rounds"] or score.get("actual_optimizer_updates")!=updates:errors.append("Frozen score training budget differs from saved training records.")
        if final.get("checkpoint_selection_changed") is True:errors.append("Test checkpoint selection changed.")
        if spec["condition"]=="s0":
            selection=final.get("identity",{}).get("selection",{})
            if train or groups or dev or updates!=0:errors.append("S0 evaluation must have zero training and no development selection.")
            if score.get("selected_round")!=0 or selection.get("selection_rule")!="predeclared_original_S0":errors.append("S0 test must use the predeclared original S0 endpoint.")
            expected=(ROOT/"runs/cafd/experiments/mistral_cafd_disjoint_v7/student_base/S0").resolve()
            checkpoint=Path(score.get("checkpoint","")).resolve()
            if checkpoint!=expected:errors.append("S0 checkpoint differs from the resolved original S0 path.")
        else:
            if len(train)!=200 or {r["round"] for r in train}!=set(range(1,201)):errors.append("Training must contain 200 unique rounds.")
            if coverage["groups"]!=800 or coverage["answers"]!=6400 or any(len(g["flags"])!=8 for g in groups.values()):errors.append("Training must contain 800 groups of eight saved answers.")
            if not 0<=updates<=200 or any(bool(r["optimizer_step"])!=active[r["round"]] for r in train):errors.append("Optimizer steps differ from active KD/RL groups; only legal skip is accepted.")
            if sorted(r["round"] for r in dev)!=MILESTONES:errors.append("Development milestone set differs.")
            for r in dev:
                saved=dev_outputs[r["round"]]
                historical_missing=spec["condition"]=="historical_mpc" and r["round"]==0 and not saved
                if not historical_missing and (r["saved_outputs"]!=64 or r["unique_ids"]!=64 or sum(x["full_pass"] for x in saved)!=r["correct"]):
                    errors.append(f"Development saved-output mismatch at round{r['round']}.")
            if dev and score.get("selected_round")!=max(dev,key=lambda r:(r["correct"],-r["round"]))["round"]:
                errors.append("Frozen model differs from development full-pass earliest-tie selection.")
    families=[]
    for family in sorted({r["family"] for r in test}):
        subset=[r for r in test if r["family"]==family]
        families.append(dict(condition=spec["condition"],family=family,correct=sum(r["full_pass"] for r in subset),
            total=len(subset),mean_reward=sum(r["reward"] for r in subset)/len(subset)))
    score=final["scores"] if final else {}
    return dict(**spec,final_available=final is not None,errors=errors,
        training_rounds=len(train),actual_optimizer_updates=updates,
        skipped_rounds=[r["round"] for r in train if not r["optimizer_step"]],
        training_answers=coverage["answers"],training_answer_tokens=coverage["tokens"],
        physical_training_generated_tokens=physical.get("training_generated_tokens",0),
        physical_training_teacher_scored_tokens=physical.get("training_teacher_scored_tokens",0),
        physical_training_reference_scored_tokens=physical.get("training_reference_scored_tokens",0),
        training_cost_counters=dict(physical),coverage=dict(coverage),development=dev,
        selected_round=score.get("selected_round"),checkpoint=score.get("checkpoint"),
        test_correct=score.get("correct"),test_total=score.get("total"),
        test_answer_tokens=sum(len(r["completion_ids"]) for r in test) if final else None,
        test_by_family=families,accounting=accounting(attempt_ids(ledger,test_ledger)),
        frozen_identity=final.get("identity") if final else None,
        _test=[dict(row_id=r["row_id"],full_pass=r["full_pass"]) for r in test])

def paired_bootstrap(condition,reference):
    a={r["row_id"]:r["full_pass"] for r in condition["_test"]}
    b={r["row_id"]:r["full_pass"] for r in reference["_test"]}
    if len(a)!=132 or a.keys()!=b.keys():raise ValueError("Paired bootstrap requires the same 132 frozen test IDs.")
    delta=np.array([int(a[k])-int(b[k]) for k in sorted(b)])
    rng=np.random.default_rng(2027)
    boot=delta[rng.integers(0,132,size=(20000,132))].mean(axis=1)
    return dict(condition=condition["condition"],reference=reference["condition"],
        difference=float(delta.mean()),bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),
        condition_only=int(sum(delta>0)),reference_only=int(sum(delta<0)),replicates=20000,seed=2027,
        paired_by="sorted frozen row_id",limitation="Single training seed; item bootstrap excludes training randomness.")

def smoke_runs(partial=False):
    base=ROOT/"artifacts/cafd/experiments"
    members=[];jobs=[]
    if base.exists():
        for art in sorted(base.glob("mistral_cafd_minimal*smoke*")):
            if not art.is_dir():continue
            ledger=list(records(art/"physical_costs.jsonl",partial))
            test=list(records(art/"final_test_physical_costs.jsonl",partial))
            ids=attempt_ids(ledger,test)
            members.append(dict(run_id=art.name,job_ids=ids))
            jobs.extend(ids)
    # Multiple smoke run IDs may share one engineering allocation; count it once.
    return [dict(run_id="minimal_baselines_engineering_smoke",included_runs=members,
        accounting=accounting(list(dict.fromkeys(jobs))))] if members else []

def mark_shared_allocations(runs,smokes):
    owners=defaultdict(list)
    for r in runs+smokes:
        for job in r["accounting"]["job_ids"]:owners[job].append(r)
    shared={}
    for job,items in owners.items():
        if len(items)>1:
            shared[job]=[r["run_id"] for r in items]
            for r in items:
                r["accounting"].update(gpu_hours=None,status="ambiguous_shared_allocation",
                    reason="Shared allocation across conditions or smoke; no unsupported cost apportionment.")
    return shared

def ready_for_final(runs,smokes,shared):
    blockers=[]
    for r in runs:
        if not r["final_available"]:blockers.append(r["condition"]+": frozen final result missing")
        blockers.extend(r["condition"]+": "+e for e in r["errors"])
        cost=r["accounting"]
        if cost["gpu_hours"] is None or not cost["terminal"]:blockers.append(r["condition"]+": terminal allocation cost unavailable")
        latest=max(cost["rows"],key=lambda a:a["start_utc"]) if cost["rows"] else None
        if r["new"] and (latest is None or latest["state"]!="COMPLETED"):
            blockers.append(r["condition"]+": latest new formal allocation is not COMPLETED")
    for r in smokes:
        if r["accounting"]["gpu_hours"] is None or not r["accounting"]["terminal"]:
            blockers.append(r["run_id"]+": smoke allocation cost incomplete")
    if shared:blockers.append("Allocation sharing prevents condition-specific cost attribution.")
    return blockers

def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n")

def write_csv(path,rows,fields):
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)

def source_proof(runs):
    paths=[Path(__file__).resolve()]
    for r in runs:
        base=ROOT/"runs/cafd/experiments"/r["run_id"]
        paths.extend(base/name for name in ["config.json","data_manifest.json","evaluation_plan.json"] if (base/name).exists())
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def render_report(runs,pairs,blockers,smokes):
    def fmt(x):return "未知" if x is None else f"{x:.6f}"
    text=["# S0、GRPO 与绝对目标 KD 最小对照","",
        "状态："+("完整结果及终态成本已核验。" if not blockers else "部分结果；等待冻结评测或终态成本。"),"",
        "| 条件 | 训练轮数 | 实际更新 | 训练回答 | 训练 tokens | 选中轮次 | test | GPU 小时 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        test="待完成" if not r["final_available"] else f"{r['test_correct']}/132"
        selected="预指定 S0" if r["condition"]=="s0" else str(r["selected_round"]) if r["selected_round"] is not None else "待完成"
        text.append(f"|{r['label']}|{r['training_rounds']}|{r['actual_optimizer_updates']}|{r['training_answers']}|{r['training_answer_tokens']}|{selected}|{test}|{fmt(r['accounting']['gpu_hours'])}|")
    text+=["","S0 在评测前固定为原始 checkpoint，0 轮训练、0 次更新，无 development 选模。其他训练条件使用 200 轮、800 组、6400 条回答预算；实际更新次数随合法 skip 如实单列，不将它们称为同更新数比较。",
        "GRPO 关闭训练 KD，但保留 control/q5 与 Teacher 诊断流程。成本包含这些诊断开销，不能解释为移除全部 Teacher 流程的最精简 GRPO 成本。",
        "绝对目标条件仅将训练 KD 目标改为 softmax((1-alpha) z_Tm + alpha z_Tnext)，保留相同 u 日程、全失败 retention 路由、全局 token 分母和 RL 项。相对 CAFD 使用永久 S0 的相对目标。",
        "","| 新条件相对已完成 relative CAFD | test 差（百分点） | 题级配对 bootstrap 95% 区间 |","|---|---:|---:|"]
    for p in pairs:text.append(f"|{p['condition']}|{p['difference']*100:.3f}|[{p['bootstrap_95'][0]*100:.3f}, {p['bootstrap_95'][1]*100:.3f}]|")
    text+=["","仅比较已经冻结并保存的 132 题 test；不会重新选模、插值或补做评测。各条件与 relative CAFD 按题 ID 配对，20,000 次 bootstrap，seed=2027。单训练 seed 的题级区间不覆盖训练随机性。",
        "这些 test 已有历史评测；development 有 Teacher 训练暴露（44/64 用于 Teacher SFT，其余 20 用于 Teacher RL）。本次是单 seed 条件比较，不能称为多 seed 因果验证。",
        "GPU 成本按各条件训练与 test 的 attempt job ID 并集去重，取 Slurm 预留终态时间；包括失败或恢复尝试。未知账目保持未知，共享作业不虚构条件分摊。已有 Teacher 轨迹训练成本不包含。Smoke 单列，不计入条件成本。",
        "","| 独立 smoke | GPU 小时 |","|---|---:|"]
    for r in smokes:text.append(f"|{r['run_id']}|{fmt(r['accounting']['gpu_hours'])}|")
    if not smokes:text.append("|未发现已记录的本组 smoke|未知|")
    if blockers:text+=["","待完成项目："]+["- "+b for b in blockers]
    text+=["","逐题型结果、development 曲线、KD/RL 覆盖、物理计数和作业终态见同目录 CSV/JSON；源码与冻结配置摘要见 source_proof.json。历史 MPC 的 round0 development 逐题输出和平均奖励未保存，保持缺失。"]
    return "\n".join(text)+"\n"

def main(argv=None):
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group();mode.add_argument("--partial",action="store_true");mode.add_argument("--final",action="store_true")
    args=parser.parse_args(argv)
    if Path.cwd().resolve()!=ROOT:raise RuntimeError("All work must remain inside CAFD.")
    runs=[summarize(spec,partial=args.partial) for spec in SPECS]
    smokes=smoke_runs(partial=args.partial)
    shared=mark_shared_allocations(runs,smokes)
    blockers=ready_for_final(runs,smokes,shared)
    reference=next(r for r in runs if r["condition"]=="relative_cafd")
    pairs=[]
    if reference["final_available"] and not reference["errors"]:
        for r in runs:
            if r["new"] and r["final_available"] and not r["errors"]:pairs.append(paired_bootstrap(r,reference))
    REPORT.mkdir(parents=True,exist_ok=True)
    public=[{k:v for k,v in r.items() if not k.startswith("_")} for r in runs]
    write_json(REPORT/"comparison.json",dict(status="complete" if not blockers else "partial",
        generated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=public,
        smoke=smokes,shared_allocations=shared,blockers=blockers,paired_bootstrap=pairs))
    write_json(REPORT/"paired_bootstrap.json",pairs)
    write_json(REPORT/"source_proof.json",source_proof(runs))
    summary=[dict(condition=r["condition"],run_id=r["run_id"],training_rounds=r["training_rounds"],
        actual_optimizer_updates=r["actual_optimizer_updates"],training_answers=r["training_answers"],
        training_answer_tokens=r["training_answer_tokens"],test_answer_tokens=r["test_answer_tokens"],
        selected_round=r["selected_round"],test_correct=r["test_correct"],test_total=r["test_total"],
        allocated_gpu_hours=r["accounting"]["gpu_hours"],allocation_status=r["accounting"]["status"]) for r in runs]
    write_csv(REPORT/"conditions.csv",summary,list(summary[0]))
    dev=[d for r in runs for d in r["development"]]
    write_csv(REPORT/"development.csv",dev,["condition","run_id","round","correct","total","mean_reward","saved_outputs","unique_ids","cumulative_allocated_gpu_hours"])
    families=[d for r in runs for d in r["test_by_family"]]
    write_csv(REPORT/"test_by_family.csv",families,["condition","family","correct","total","mean_reward"])
    cov=[dict(condition=r["condition"],**r["coverage"]) for r in runs]
    write_csv(REPORT/"coverage.csv",cov,list(cov[0]))
    allocations=[dict(condition=r.get("condition","smoke"),run_id=r["run_id"],**a) for r in runs+smokes for a in r["accounting"]["rows"]]
    write_csv(REPORT/"allocations.csv",allocations,["condition","run_id","job_id","state","start_utc","end_utc","elapsed_seconds","gpus","allocation","node","exit_code"])
    (REPORT/"MINIMAL_BASELINES_FINAL.md").write_text(render_report(runs,pairs,blockers,smokes))
    print(json.dumps(dict(status="complete" if not blockers else "partial",frozen_results=sum(r["final_available"] for r in runs),
        required_results=5,blockers=blockers,report=str(REPORT/"MINIMAL_BASELINES_FINAL.md")),ensure_ascii=False))
    return 0 if args.partial or not blockers else 2

if __name__=="__main__":raise SystemExit(main())
