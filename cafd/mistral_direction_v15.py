"""Fit-only, no-generation paired target diagnostic; no optimizer or test reads."""
import gc
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from .canonical import canonical_program, fenced
from .verifier import score_completion
from .mistral_runtime import load_tokenizers, assert_exact_tokenizer_pair
from .mistral_target_audit_v13 import load_hidden, log_probs, sample_positions, write_json

ROOT = Path("/blue/du.j/jinjiaguo/CAFD")
RUN = "mistral_cafd_direction_v15"

def choose(path, phase, fit, count=8):
    chosen = {}
    with path.open() as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("support_phase") != phase or item.get("rollout_source") != "student_on_policy":
                continue
            key = item["row_id"]
            if key not in fit:
                raise RuntimeError("non-fit rollout encountered")
            if key not in chosen and item["completion_ids"]:
                chosen[key] = item
            if len(chosen) == count:
                break
    if len(chosen) != count:
        raise RuntimeError("insufficient persisted student rollouts")
    return list(chosen.values())

def targets(ref, prev, nxt):
    delta = nxt - prev
    residual = ref - prev
    d = (delta - delta.mean(-1, keepdim=True)).square().mean(-1).sqrt()
    r = (residual - residual.mean(-1, keepdim=True)).square().mean(-1).sqrt()
    alpha = torch.where(d+r > 0, d/(d+r).clamp_min(1e-30), torch.zeros_like(d))
    relative = F.log_softmax(ref + delta, -1)
    mix = torch.logaddexp(torch.log1p(-alpha)[:, None] + nxt,
                         alpha.log()[:, None] + relative)
    return relative, mix, alpha

@torch.inference_mode()
def main():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("wrong working directory")
    start = time.monotonic()
    out = ROOT / "artifacts/cafd/experiments" / RUN
    if (out / "diagnosis.json").exists():
        raise RuntimeError("completed diagnostic already exists; refusing overwrite")
    fit = {x["id"]: x for x in map(json.loads, (ROOT/"data/cafd/generalization_v10/fit.jsonl").open())}
    assert len(fit) == 614
    teacher_tok, tok = load_tokenizers(ROOT/".cache/huggingface/hub")
    assert_exact_tokenizer_pair(teacher_tok, tok)
    base = ROOT/"runs/cafd/experiments/mistral_cafd_disjoint_v7"
    old = ROOT/"runs/cafd/experiments/mistral_cafd_gen_v10_mixed/cafd"
    new = ROOT/"runs/cafd/experiments/mistral_cafd_anchored_v14/cafd"
    route = json.loads((base/"teacher/route.json").read_text())
    assert route["status"] == "frozen" and len(route["checkpoints"]) == 6
    paths = [Path(x["checkpoint"]) for x in route["checkpoints"]]
    device = torch.device("cuda:0")
    records = []
    manifest = []
    endpoint = load_hidden(ROOT, paths[-1], device)
    for phase in (0, 1):
        items = choose(old/"raw_rollouts.jsonl", phase, fit)
        previous = load_hidden(ROOT, paths[phase], device)
        nxt = load_hidden(ROOT, paths[phase+1], device)
        references = [("original", base/"student_base/S0" if phase == 0 else old/"step40")]
        if phase == 1:
            references.append(("anchored", new/"step40"))
        for name, path in references:
            ref = load_hidden(ROOT, path, device)
            for item in items:
                row = fit[item["row_id"]]
                gold_text = fenced(canonical_program(row))
                gold_ids = tok.encode(gold_text, add_special_tokens=False)
                assert len(gold_ids)+1 <= 2048, "gold truncation"
                assert score_completion(tok.decode(gold_ids, skip_special_tokens=True), row["ground_truth"], "full_pass", require_contract=True) == 1.0
                student_pass = score_completion(tok.decode(item["completion_ids"], skip_special_tokens=True), row["ground_truth"], "full_pass", require_contract=True)
                manifest.append({"phase":phase,"reference":name,"id":row["id"],"checkpoint":str(path),
                                 "gold_tokens":len(gold_ids),"gold_pass":True,"student_pass":student_pass,
                                 "source_update":item["update"]})
                for source, completion in (("verified_gold", gold_ids+[tok.eos_token_id]), ("persisted_student", item["completion_ids"])):
                    record = dict(item, completion_ids=completion)
                    ids, attention, positions, labels = sample_positions(record, tok.eos_token_id, device, 128)
                    lr = log_probs(ref, ids, attention, positions)
                    lp = log_probs(previous, ids, attention, positions)
                    ln = log_probs(nxt, ids, attention, positions)
                    le = log_probs(endpoint, ids, attention, positions)
                    lq, lm, alpha = targets(lr, lp, ln)
                    row_indices = torch.arange(len(labels), device=device)
                    nll = {k: -v[row_indices,labels] for k,v in
                           (("reference",lr),("previous_teacher",lp),("next_teacher",ln),
                            ("endpoint",le),("relative",lq),("anchored",lm))}
                    # Only literal fences/EOS are classified as wrapper. DSL body is NOT labelled semantic correctness.
                    offsets = positions.cpu().tolist()
                    completion_offsets = [p+1-len(item["prompt_ids"]) for p in offsets]
                    for i, token_offset in enumerate(completion_offsets):
                        prefix = tok.decode(completion[:token_offset], skip_special_tokens=True)
                        through = tok.decode(completion[:token_offset+1], skip_special_tokens=True)
                        emitted = through[len(prefix):] if through.startswith(prefix) else ""
                        wrapper = labels[i].item() == tok.eos_token_id or "`" in emitted or (token_offset < 8 and "manufactoria" in through)
                        records.append({"phase":phase,"reference":name,"row_id":row["id"],
                            "source":source,"student_full_pass":student_pass,"position":token_offset,
                            "token_group":"wrapper_proxy" if wrapper else "body_or_other",
                            "alpha":alpha[i].item(), **{k+"_nll":v[i].item() for k,v in nll.items()}})
                    del lr, lp, ln, le, lq, lm, alpha, nll, ids, attention
                write_json(out/"progress.json", {"status":"running","phase":phase,"reference":name,"completed_pairs":len(manifest)})
            del ref
            gc.collect()
            torch.cuda.empty_cache()
        del previous, nxt
        gc.collect()
        torch.cuda.empty_cache()
    groups = []
    for phase in (0,1):
        for reference in ("original","anchored"):
            for source in ("verified_gold","persisted_student"):
                selected = [x for x in records if x["phase"]==phase and x["reference"]==reference and x["source"]==source]
                if not selected: continue
                # Equal task weighting; token counts are not independent samples.
                tasks = []
                for rid in sorted({x["row_id"] for x in selected}):
                    rows = [x for x in selected if x["row_id"]==rid]
                    tasks.append({k:sum(x[k] for x in rows)/len(rows) for k in rows[0] if k.endswith("_nll") or k=="alpha"})
                groups.append({"phase":phase,"reference":reference,"source":source,"tasks":len(tasks),
                    "sampled_positions":len(selected),
                    **{k:sum(x[k] for x in tasks)/len(tasks) for k in tasks[0]},
                    "tasks_relative_better_than_ref":sum(x["relative_nll"]<x["reference_nll"] for x in tasks),
                    "tasks_anchored_better_than_relative":sum(x["anchored_nll"]<x["relative_nll"] for x in tasks)})
    write_json(out/"diagnosis.json", {"status":"complete","groups":groups,"manifest":manifest,
        "tokens":records,"elapsed_seconds":time.monotonic()-start,"gpu":torch.cuda.get_device_name(),
        "selection_accessed":False,"confirmation_accessed":False,"frozen_test_accessed":False,
        "limitations":["8 first distinct fit prompts per phase; deterministic nonrandom diagnostic sample",
        "128 evenly sampled positions maximum; not full-sequence likelihood",
        "Student tokens are not gold; their NLL is not accuracy",
        "canonical-prefix NLL does not certify correctness on divergent Student prefixes",
        "wrapper proxy is not a semantic-token annotation","no causal or generalization conclusion"]})
    print(json.dumps({"status":"complete","groups":groups,"elapsed_seconds":time.monotonic()-start}), flush=True)

if __name__ == "__main__":
    main()
