"""Two-GPU inference-only execution of frozen T1/T2. Never starts training."""
from __future__ import annotations
import argparse
import datetime as dt
import gc
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path("/blue/du.j/jinjiaguo/CAFD")
RUN = ROOT/"runs/cafd/experiments/mistral_target_path_t1t2_v1"


def read_json(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def publish(path, value):
    from .target_path_t2 import write_json
    write_json(path, value)


def prepare_plan():
    manifest = read_json(RUN/"manifest.json")
    selection = rows(RUN/"selection.jsonl")
    route = manifest["route"]
    s0 = manifest["student_base"]["checkpoint"]
    t0 = route[0]["checkpoint"]
    M = len(route)-1
    policies = {}
    for m,item in enumerate(route):
        q = [(s0,1)] if m == 0 else [(s0,1),(item["checkpoint"],1),(t0,-1)]
        for prefix,terms in (("q",q),("p",[(item["checkpoint"],1)])):
            policies[f"{prefix}{m}_random"] = dict(terms=terms, sample=True, count=2)
    for name in ("q0",f"q{M}",f"p{M}"):
        policies[name+"_greedy"] = dict(policies[name+"_random"], sample=False,count=1)
    plan = dict(M=M, policies=policies, selection_ids=[r["id"] for r in selection],
        seed_table={str(r["id"]):[2027+100000*j+i for j in range(2)] for i,r in enumerate(selection)},
        decoding=dict(batch_size=8,max_new_tokens=2048,max_prompt_tokens=4096,
                      temperature=1.,top_p=1.,top_k=0),
        queue_order=[f"q{m}_random" for m in range(M,-1,-1)] + [f"p{m}_random" for m in range(M,-1,-1)]
            + [f"q{M}_greedy",f"p{M}_greedy","q0_greedy"],
        max_gpus=2, reserved_memory_gib=192, training_submitted=False, frozen_test_accessed=False)
    path=RUN/"execution_plan.json"
    if path.exists() and read_json(path)!=plan:
        raise RuntimeError("frozen diagnostic plan changed")
    publish(path,plan)
    publish(RUN/"t2_model_map.json",dict(s0=s0,M=M,
        teacher_route=[dict(stage=x["index"],path=x["checkpoint"]) for x in route]))
    return plan


def run_t2(pool, tokenizer, plan):
    import torch
    from .target_path_t1 import run_policy
    from .target_path_prepare import prediction_positions
    from .target_path_t2 import run
    tasks=rows(RUN/"t2_tasks.jsonl")
    wanted={r["row_id"] for r in tasks}
    fit=rows(ROOT/"data/cafd/generalization_v10/fit.jsonl")
    task_by_id={r["row_id"]:r for r in tasks}
    fit_by_id={r["id"]:r for r in fit}
    ordered=[fit_by_id[t["row_id"]] for t in tasks]
    s0=read_json(RUN/"manifest.json")["student_base"]["checkpoint"]
    run_policy(ROOT,RUN,"t2_s0_prefix",[(s0,1)],ordered,
        dict(plan["decoding"],sample=True,count=1),
        {t["row_id"]:[302027+i] for i,t in enumerate(tasks)}, "cuda:0",pool=pool,tokenizer=tokenizer)
    existing=rows(RUN/"t2_prefixes.jsonl")
    additions=[]
    for r in rows(RUN/"policies/t2_s0_prefix/rollouts.jsonl"):
        if r["status"]=="ok" and r["completion_ids"]:
            task=task_by_id[r["row_id"]]
            additions.append(dict(**task,rowid=r["row_id"],source="S0",checkpoint=s0,
                prompt_ids=r["prompt_ids"],completion_ids=r["completion_ids"],
                positions=prediction_positions(r["completion_ids"]),
                provenance=dict(source="t2_s0_prefix",seed=r["seed"],top_p=1.,temperature=1.,
                    output_stop="external_closing_fence_or_generated_EOS_no_synthetic_EOS",
                    checkpoint_weights_preserved=True)))
    combined=[r for r in existing if r["source"]!="S0"]+additions
    path=RUN/"t2_all_prefixes.json"
    if path.exists() and read_json(path)!=combined:
        raise RuntimeError("fixed T2 prefix identity changed")
    publish(path,combined)
    pool.close()
    gc.collect()
    torch.cuda.empty_cache()
    result=run(argparse.Namespace(manifest=path,model_map=RUN/"t2_model_map.json",
        output_dir=RUN/"t2",device="cuda:0",position_chunk=32,skip_scoring=False,
        legacy_cache_proof=RUN/"manifest.json"))
    publish(RUN/"t2_done.json",dict(status=result["status"],finished_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    return result


def worker(index):
    import torch
    from .mpc_allocation import validate_allocation
    from .mistral_runtime import load_tokenizers, assert_exact_tokenizer_pair
    from .prompting import assert_matches_mistral_chat_template
    from .target_path_t1 import EnsemblePool, run_policy
    os.chdir(ROOT)
    plan=read_json(RUN/"execution_plan.json")
    if int(os.environ.get("SLURM_GPUS_ON_NODE","0"))>2:
        raise RuntimeError("task prohibits more than two allocated GPUs")
    validate_allocation(os.environ,torch.cuda.get_device_name(0),torch.cuda.device_count())
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4)
    start=time.time()
    status=dict(worker=index,job_id=os.environ.get("SLURM_JOB_ID"),node=os.environ.get("HOSTNAME"),
        pid=os.getpid(),started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),start_unix=start,
        gpu=torch.cuda.get_device_name(0),cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        training_updates=0,status="running")
    publish(RUN/f"worker{index}_state.json",status)
    teacher_tok,student_tok=load_tokenizers(ROOT/".cache/huggingface/hub")
    pair=assert_exact_tokenizer_pair(teacher_tok,student_tok)
    renderer=assert_matches_mistral_chat_template(student_tok)
    publish(RUN/f"worker{index}_preflight.json",dict(tokenizer_pair_passed=True,renderer_passed=True,vocabulary=131072,device=status["gpu"]))
    pool=EnsemblePool(ROOT,"cuda:0",student_tok)
    selection=rows(RUN/"selection.jsonl")
    failures=[]
    def policy(name, maximum=None):
        spec=plan["policies"][name]
        status["stage"]=name
        publish(RUN/f"worker{index}_state.json",status)
        return run_policy(ROOT,RUN,name,spec["terms"],selection,
            dict(plan["decoding"],sample=spec["sample"],count=spec["count"]),
            plan["seed_table"],"cuda:0",pool=pool,tokenizer=student_tok,max_new_records=maximum)
    try:
        pilots=["q0_random","q5_random"] if index==0 else ["p5_random"]
        for name in pilots:
            claim=RUN/"claims"/name
            claim.parent.mkdir(exist_ok=True)
            claim.mkdir(exist_ok=True)
            publish(claim/"owner.json",dict(worker=index,pid=os.getpid(),job=status["job_id"]))
            try:
                result=policy(name,8)
                publish(RUN/f"pilot_{name}.json",result)
            except Exception as exc:
                failures.append(dict(stage=name,pilot=True,error=repr(exc),traceback=traceback.format_exc()))
        if index==0:
            try:
                status["stage"]="T2_prefix_scoring_and_geometry"
                publish(RUN/f"worker{index}_state.json",status)
                run_t2(pool,student_tok,plan)
            except Exception as exc:
                failures.append(dict(stage="T2",error=repr(exc),traceback=traceback.format_exc()))
                publish(RUN/"t2_driver_error.json",failures[-1])
        # p5 pilot is already resident on worker1; finish it first. Thereafter
        # atomic per-policy claims balance the remaining work without collisions.
        order=(["p5_random"] if index==1 else [])+plan["queue_order"]
        for name in dict.fromkeys(order):
            claim=RUN/"claims"/name
            claim.parent.mkdir(exist_ok=True)
            try:
                claim.mkdir()
            except FileExistsError:
                owner=read_json(claim/"owner.json") if (claim/"owner.json").exists() else {}
                if owner.get("worker") != index or owner.get("pid") != os.getpid():
                    continue
            publish(claim/"owner.json",dict(worker=index,pid=os.getpid(),job=status["job_id"]))
            try:
                result=policy(name)
                publish(claim/"done.json",result)
            except Exception as exc:
                failures.append(dict(stage=name,error=repr(exc),traceback=traceback.format_exc()))
                publish(claim/"error.json",failures[-1])
    finally:
        pool.close()
        status.update(status="complete" if not failures else "partial",failures=failures,
            finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=time.time()-start)
        publish(RUN/f"worker{index}_state.json",status)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prepare",action="store_true")
    p.add_argument("--worker",type=int,choices=[0,1])
    args=p.parse_args()
    if args.prepare:
        print(json.dumps(prepare_plan(),indent=2))
    elif args.worker is not None:
        worker(args.worker)
    else:
        p.error("choose --prepare or --worker")
if __name__=="__main__":
    main()
