"""Audit fully completed T1/T2 artifacts without generation or test access.

Partial runs are reported by target_path_report, not certified complete here.
"""
import argparse
from collections import Counter,defaultdict
import csv
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def validate(run):
    run=Path(run)
    selection=lines(run/"selection.jsonl")
    ids={r["id"] for r in selection}
    assert len(ids)==64
    plan=read(run/"execution_plan.json")
    records=lines(run/"target_rollouts.jsonl")
    assert len(records)==1728
    groups=defaultdict(list)
    for r in records:
        groups[r["policy_id"]].append(r)
    assert set(groups)==set(plan["policies"])
    findings=[]
    for policy,rows in groups.items():
        spec=plan["policies"][policy]
        count=spec["count"]
        expected={(i,j) for i in ids for j in range(count)}
        observed=[(r["row_id"],r["sample_index"]) for r in rows]
        assert len(set(observed))==len(observed) and set(observed)==expected
        actual=[r for r in rows if r["status"]!="missing"]
        for r in actual:
            assert r["seed"]==plan["seed_table"][r["row_id"]][r["sample_index"]]
            assert r["generated_tokens"]==len(r["completion_ids"])<=2048
            assert len(r["prompt_ids"])<=4096
            assert all(0<=t<131072 for t in r["prompt_ids"]+r["completion_ids"])
            assert len(r["selected_log_probs"])==r["generated_tokens"]
            assert all(math.isfinite(x) and x<=1e-6 for x in r["selected_log_probs"])
            if r["status"]=="ok":
                assert r["full_pass"]==bool(r["verifier"].get("all_passed",False))
                assert r["full_pass"]==(r["reward"]==1.)
            if r["stop_reason"]=="eos":
                assert r["completion_ids"][-1]==2
            if r["truncated"]:
                assert r["generated_tokens"]==2048 and r["stop_reason"]=="max_new_tokens"
        cost=read(run/"policies"/policy/"costs.json")
        assert sum(r["generated_tokens"] for r in actual)==cost["generated_tokens"]
        findings.append(dict(policy_id=policy,planned=len(rows),recorded=len(actual),
            errors=sum(r["status"]=="error" for r in actual),
            successes=sum(r["full_pass"] for r in actual)))
    profiles=list(csv.DictReader((run/"target_profile.csv").open()))
    assert len(profiles)==60
    expected_profile={(p,f) for p in groups for f in {"all"}|{r["problem_family"] for r in selection}}
    profile_keys=[(p["policy_id"],p["family"]) for p in profiles]
    assert len(set(profile_keys))==len(profile_keys) and set(profile_keys)==expected_profile
    for p in profiles:
        subset=[r for r in groups[p["policy_id"]] if p["family"]=="all" or r["family"]==p["family"]]
        assert int(p["N"])==len(subset)
        assert int(p["correct"])==sum(r["full_pass"] for r in subset)
    prefixes=read(run/"t2_all_prefixes.json")
    sources=Counter(r["source"] for r in prefixes)
    assert sources=={"S0":32,"early_student":32,"late_student":32}
    task_split={}
    prefix_keys=[(r["source"],r.get("rowid",r.get("row_id"))) for r in prefixes]
    assert len(prefix_keys)==len(set(prefix_keys))
    assert len({key[1] for key in prefix_keys})==32
    for r in prefixes:
        key=r.get("rowid",r.get("row_id"))
        assert task_split.setdefault(key,r["split"])==r["split"]
        assert len(r["positions"])<=16 and len(set(r["positions"]))==len(r["positions"])
        assert all(0<=i<len(r["completion_ids"]) for i in r["positions"])
    assert Counter(task_split.values())=={"calibration":16,"heldout":16}
    geometry=read(run/"t2/path_geometry.json")
    assert geometry["status"]=="complete" and not geometry["failures"]
    assert len(geometry["stage_results"])==24
    geometry_keys=[(r["stage"],r["source"]) for r in geometry["stage_results"]]
    expected_geometry={(m,s) for m in range(6) for s in {"all","S0","early_student","late_student"}}
    assert len(set(geometry_keys))==len(geometry_keys) and set(geometry_keys)==expected_geometry
    shared_a={}
    for row in geometry["stage_results"]:
        assert row["fixed_a"]==row["stage"]/5.
        assert 0<=row["fitted_a"]<=1.
        assert shared_a.setdefault(row["stage"],row["fitted_a"])==row["fitted_a"]
        if row["stage"] in (0,5):
            assert all(row[key]==0. for key in ("fixed_calibration_kl","fixed_heldout_kl",
                "fitted_calibration_kl","fitted_heldout_kl"))
        for key,value in row.items():
            if key.endswith("_kl") and value is not None:
                assert math.isfinite(value) and value>=-1e-5
    costs=read(run/"costs.json")
    assert costs["optimizer_updates"]==0 and costs["frozen_test_read"] is False
    assert costs["t1"]["generated_tokens"]==sum(r["generated_tokens"] or 0 for r in records)
    assert costs["verified_loadability"]["verified_count"]==7
    for name in ("cumulative_relative","absolute_teacher"):
        cfg=read(run/"configs"/f"{name}.json")
        assert cfg["status"]=="PREPARED_NOT_SUBMITTED"
        assert not Path(cfg["output_directory"]).exists()
    result=dict(status="passed" if all(r["recorded"]==r["planned"] and r["errors"]==0 for r in findings) else "partial",
        t1=findings,t1_planned=1728,t2_sources=dict(sources),t2_positions=sum(len(r["positions"]) for r in prefixes),
        t2_stages=6,actual_models_verified=7,optimizer_updates=0,frozen_test_read=False,
        verified=["episode_grid","seed_pairing","token_and_log_probability_lengths",
            "official_verifier_consistency","unchanged_denominators","generation_caps",
            "task_disjoint_calibration","geometry_endpoints","cost_token_totals","KD_not_started"])
    (run/"artifact_validation.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(validate(args.run_dir),indent=2))
