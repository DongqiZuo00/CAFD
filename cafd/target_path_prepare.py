"""Freeze T1/T2 inputs without reading test data, weights, or target scores.

Historical Student prefix source identity is a training run plus the exact
pre-update step. A missing on-disk checkpoint is never replaced by a nearby one.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .mpc_data import (FAMILIES, MODEL_IDENTITIES, ROUTE, SOURCE, STUDENT_BASE,
                       TEACHER_RUN, TEACHER_SPLIT, _inside, _json, _rows,
                       _publish_unchanged, _pretty, checkpoint_metadata)

OLD_RUN = Path("runs/cafd/experiments/mistral_cafd_gen_v10_mixed/cafd")


def prediction_positions(tokens, eos_id=2, maximum=16):
    """Stratified valid completion-token offsets, including first generated EOS."""
    valid = len(tokens)
    if eos_id in tokens:
        valid = tokens.index(eos_id) + 1
    count = min(maximum, valid)
    if count <= 0:
        return []
    return [min(valid - 1, int((i + .5) * valid / count)) for i in range(count)]


def stratified_tasks(rows, seed=2027, maximum=32, quotas=None):
    """Fixed balanced family sample and within-family split, before scoring."""
    rng = random.Random(seed)
    selected = []
    families = sorted({r["problem_family"] for r in rows})
    if not families or maximum < len(families):
        raise ValueError("insufficient tasks for stratification")
    for index, family in enumerate(families):
        candidates = sorted((r for r in rows if r["problem_family"] == family), key=lambda r: r["id"])
        rng.shuffle(candidates)
        quota = quotas[family] if quotas is not None else maximum // len(families) + int(index < maximum % len(families))
        chosen = candidates[:quota]
        # Odd quotas alternate the extra item so 11/11/10 gives 16/16.
        calibration_count = len(chosen)//2 + int(len(chosen)%2 and index%2==1)
        selected.extend(dict(row_id=r["id"], family=family,
                             split="calibration" if j < calibration_count else "heldout")
                        for j,r in enumerate(chosen))
    if len({r["row_id"] for r in selected}) != len(selected):
        raise ValueError("duplicate input IDs")
    return selected


def shared_prefix_cohort(root, fit):
    """Use only ID/source/update metadata, never rewards or target scores."""
    source=_inside(root,OLD_RUN/"raw_rollouts.jsonl")
    early,late=set(),set()
    if source.exists():
        for line in source.open(encoding="utf-8"):
            row=json.loads(line)
            if row.get("rollout_source") != "student_on_policy":
                continue
            if 2 <= int(row["update"]) <= 40:
                early.add(row["row_id"])
            if 161 <= int(row["update"]) <= 200:
                late.add(row["row_id"])
    rows=[r for r in fit if r["id"] in early&late]
    counts=Counter(r["problem_family"] for r in rows)
    quotas={"contains_count":5,"contains_ordered":13,"contains_substring":14}
    if all(counts[k]>=n for k,n in quotas.items()):
        tasks=stratified_tasks(rows,quotas=quotas)
        return tasks,dict(kind="coverage_selected_early_late_shared_fit_cohort",
            eligible_count=len(rows),eligible_family_counts=dict(counts),quotas=quotas,
            population_unbiased_claim=False,reward_or_target_scores_used=False)
    return stratified_tasks(fit),dict(kind="full_fit_stratified_fallback",eligible_count=len(rows),
        eligible_family_counts=dict(counts),reason="shared_cohort_cannot_fill_frozen_family_quotas",
        reward_or_target_scores_used=False)


def _checkpoint(root, path, **identity):
    result = dict(checkpoint=str(_inside(root,path)), **identity)
    try:
        result.update(status="metadata_valid", metadata=checkpoint_metadata(root,path))
    except (OSError,ValueError,KeyError) as error:
        result.update(status="missing_or_invalid", error=f"{type(error).__name__}: {error}")
    result["actual_model_loaded"] = False
    return result


def historic_prefixes(root, tasks):
    path = _inside(root, OLD_RUN/"raw_rollouts.jsonl")
    selected = {r["row_id"]:r for r in tasks}
    found = {}
    checkpoint_cache = {}
    if not path.is_file():
        return [], [dict(source=s, missing_ids=list(selected), reason="raw_rollouts_missing")
                    for s in ("S0", "early_student", "late_student")]
    for line_number, line in enumerate(path.open(encoding="utf-8"), 1):
        record = json.loads(line)
        if record.get("rollout_source") != "student_on_policy" or record.get("row_id") not in selected:
            continue
        step = int(record["update"]) - 1
        source = "S0" if step == 0 else "early_student" if 1 <= step <= 39 else "late_student" if 160 <= step <= 199 else None
        key = (source,record["row_id"])
        if source is None or key in found:
            continue
        prompt, completion = record.get("prompt_ids"),record.get("completion_ids")
        if not isinstance(prompt,list) or not isinstance(completion,list) or not prompt or not completion:
            continue
        if any(type(t) is not int or not 0 <= t < 131072 for t in prompt+completion):
            raise ValueError(f"bad historical tokens at line {line_number}")
        checkpoint = _inside(root, STUDENT_BASE if step == 0 else OLD_RUN/f"step{step}")
        # Never call step40 the source of an update40 rollout (its source is step39).
        if str(checkpoint) not in checkpoint_cache:
            try:
                checkpoint_metadata(root,checkpoint)
                checkpoint_cache[str(checkpoint)]=str(checkpoint)
            except (OSError,ValueError,KeyError):
                checkpoint_cache[str(checkpoint)]=None
        checkpoint=checkpoint_cache[str(checkpoint)]
        task = selected[record["row_id"]]
        found[key] = dict(**task, rowid=record["row_id"], source=source, checkpoint=checkpoint,
            prompt_ids=prompt, completion_ids=completion,
            positions=prediction_positions(completion),
            provenance=dict(source_run=str(_inside(root,OLD_RUN)), source_file=str(path),
                source_line=line_number, raw_update=int(record["update"]), behavior_after_update=step,
                checkpoint_weights_preserved=checkpoint is not None,
                behavior_identity=f"mistral_cafd_gen_v10_mixed/cafd@after_update_{step}",
                decoder_config="configs/cafd/manufactoria_mistral_cafd_gen_v10_mixed.yaml",
                temperature=1., top_p=.95, max_new_tokens=2048,
                output_stop="historical_closing_fence_forces_EOS_or_generated_EOS",
                sample_slot=record.get("sample_slot"), prompt_slot=record.get("prompt_slot"),
                source_is_not_current_fixed_KD_training=True))
    missing = [dict(source=s, missing_ids=[r["row_id"] for r in tasks if (s,r["row_id"]) not in found],
                    reason="no_historical_student_rollout_for_fixed_task_and_source")
               for s in ("S0", "early_student", "late_student")]
    return list(found.values()),missing


def kd_config(root, run_dir, target, route, fit, selection):
    from .data import prompt_stream
    m = len(route)-1
    stream = prompt_stream(fit,200,4,2027)
    return dict(protocol="CAFD-fixed-target-pure-KD-v1", status="PREPARED_NOT_SUBMITTED",
        target=target, root=str(root), diagnostic_run=str(run_dir),
        output_directory=str(run_dir/f"kd_{target}"), student_base=str(_inside(root,STUDENT_BASE)),
        route=[r["checkpoint"] for r in route], seed=2027, total_updates=200,
        u_by_update=[m*(segment+1)/5 for segment in range(5) for _ in range(40)],
        milestones=[0,10,40,80,120,160,200], learning_rate=1e-6,
        optimizer="FP32-master-AdamW", betas=[.9,.999], eps=1e-8, weight_decay=0.,
        lr_schedule="constant", grad_clip=1., token_chunk=64,
        prompts_per_update=4, rollouts_per_prompt=8,
        max_new_tokens=2048, max_prompt_tokens=4096,
        temperature=1.,top_p=1.,top_k=0, sample=True,
        output_stop="external_closing_fence_or_generated_EOS_no_synthetic_EOS",
        support="current_student_only", loss="exact_full_vocabulary_forward_KL",
        normalization="all_valid_completion_tokens_across_32_slots",
        reference="permanent_initial_S0" if target=="cumulative_relative" else None,
        rl_loss=False, reward_routing=False, mpc=False, adaptive_schedule=False,
        training_data=str(_inside(root,SOURCE/"fit.jsonl")), training_ids=[r["id"] for r in fit],
        development_data=str(run_dir/"selection.jsonl"),development_ids=[r["id"] for r in selection],
        prompt_stream=[dict(update=i+1,indices=indices,ids=[fit[j]["id"] for j in indices])
                       for i,indices in enumerate(stream)],
        selection_rule="development_full_pass_earliest_tie", frozen_test_accessed=False,
        maximum_gpus=2,runner_gpus=1,reserved_memory_gib=192,
        estimated_additional_disk_gib_per_method=100,
        resource_estimate="One B200 serial runner; timing estimate must use real diagnostic/first-update throughput.")


def prepare(root, run_dir):
    root=Path(root).resolve()
    run_dir=_inside(root,run_dir)
    if not run_dir.is_relative_to(root/"runs/cafd/experiments"):
        raise ValueError("new diagnostic run must be inside runs/cafd/experiments")
    route=_json(_inside(root,ROUTE))
    if route.get("status") != "frozen":
        raise ValueError("Teacher route is not frozen")
    records=[_checkpoint(root,item["checkpoint"],index=i,step=item["step"])
             for i,item in enumerate(route["checkpoints"])]
    student=_checkpoint(root,STUDENT_BASE,index=None,step="raw_instruct_S0_NOT_capacity_SFT")
    fit=_rows(_inside(root,SOURCE/"fit.jsonl"))
    selection=_rows(_inside(root,SOURCE/"selection.jsonl"))
    original=_json(_inside(root,SOURCE/"manifest.json"))
    if [r["id"] for r in fit] != original["fit_ids"] or [r["id"] for r in selection] != original["selection_ids"]:
        raise ValueError("frozen source IDs changed")
    if len(fit)!=614 or len(selection)!=64 or set(original["fit_ids"])&set(original["selection_ids"]):
        raise ValueError("invalid fit/selection counts or overlap")
    teacher_split=_json(_inside(root,TEACHER_SPLIT))
    sft=set(teacher_split["teacher_sft_ids"])
    rl=set(teacher_split["teacher_rlvr_ids"])
    selection_ids={r["id"] for r in selection}
    tasks,cohort=shared_prefix_cohort(root,fit)
    prefixes,missing=historic_prefixes(root,tasks)
    # T1 is freshly generated; only T2 may reuse these differently decoded prefixes.
    manifest=dict(protocol="CAFD-target-path-T1T2-v1",status="frozen",seed=2027,
        benchmark="DELTA Manufactoria-HAS",models=MODEL_IDENTITIES,route=records,
        route_source=str(_inside(root,ROUTE)),route_kind=route["kind"],student_base=student,
        student_is_capacity_checkpoint=False,
        other_teacher_capacity_directories=[str(p) for p in sorted((_inside(root,TEACHER_RUN/"teacher_capacity")).glob("SFT*")) if p.is_dir()],
        selection=dict(path=str(run_dir/"selection.jsonl"),source=str(_inside(root,SOURCE/"selection.jsonl")),
            ids=[r["id"] for r in selection],family_counts=dict(Counter(r["problem_family"] for r in selection)),
            teacher_sft_seen_ids=sorted(selection_ids&sft),teacher_rl_seen_ids=sorted(selection_ids&rl),
            teacher_sft_overlap_count=len(selection_ids&sft),teacher_rl_overlap_count=len(selection_ids&rl),
            previous_student_selection_exposure=True,independent_generalization_claim=False),
        confirmation=dict(source="data/cafd/development.jsonl",historically_used_for_teacher_selection_and_student_observation=True,
            evidence=["runs/cafd/experiments/mistral_cafd_disjoint_v7/teacher/gate.json",str(OLD_RUN/"confirmation.json")],
            read_this_run=False,used_this_run=False),
        test=dict(read_this_run=False,evaluated_this_run=False,results_used=False),
        decoding=dict(temperature=1.,top_p=1.,top_k=0,max_new_tokens=2048,
            max_prompt_tokens=4096,stochastic_per_task=2,greedy_per_task=1,
            greedy_policies=["S0","q_M","p_TM"],full_vocabulary=True,
            seed_table_rule="2027 + 100000*sample_index + task_index; same across policies",
            eos_id=2,output_stop="external_closing_fence_or_generated_EOS_no_synthetic_EOS"),
        t2=dict(source=str(_inside(root,SOURCE/"fit.jsonl")),tasks=tasks,
            cohort=cohort,
            task_count=len(tasks),seed=2027,max_positions_per_completion=16,
            position_definition="zero_based_completion_prediction_offset; includes_first_EOS_excludes_after_EOS",
            sampling_before_target_scores=True,prefix_count=len(prefixes),missing=missing,
            historical_decoder_is_different_from_T1=True,
            source_selection="first complete Student rollout in each predeclared source window, independent of target scores"),
        missing_checkpoint_policy="continue every independent available diagnostic; do not train or replace missing nodes",
        hashes_computed=False,maximum_gpus=2,maximum_memory_gib=192)
    run_dir.mkdir(parents=True,exist_ok=True)
    for filename, rows in (("selection.jsonl",selection),("t2_tasks.jsonl",tasks),("t2_prefixes.jsonl",prefixes)):
        _publish_unchanged(run_dir/filename,"".join(json.dumps(r,sort_keys=True,ensure_ascii=False)+"\n" for r in rows))
    for target in ("cumulative_relative","absolute_teacher"):
        cfg=kd_config(root,run_dir,target,records,fit,selection)
        (run_dir/"configs").mkdir(exist_ok=True)
        _publish_unchanged(run_dir/"configs"/f"{target}.json",_pretty(cfg))
    _publish_unchanged(run_dir/"manifest.json",_pretty(manifest))
    return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,required=True)
    args=parser.parse_args()
    result=prepare(args.root,args.run_dir)
    print(json.dumps({"route":{str(r["index"]):r["status"] for r in result["route"]},
                      "t2_prefixes":result["t2"]["prefix_count"],"test_read":False},sort_keys=True))
