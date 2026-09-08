"""Read-only target-path geometry on predeclared Student prefixes.

No generation, optimizer, development selection, or test-data access occurs here.
FP32 full-vocabulary logits are cached once per checkpoint; coefficient fitting
never reruns a Transformer. All reported KLs retain their signed floating value.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path("/blue/du.j/jinjiaguo/CAFD")


class PhysicalAttemptLedger:
    """Append-only physical resources, separate from overwritten result snapshots.

    Time is added incrementally at completed operation boundaries, including
    failed operations. An interrupted last operation is explicitly a lower bound
    on subsequent replay. Cache metadata never adds cost a second time.
    """
    def __init__(self, output, clock=time.monotonic):
        from .mpc_ledger import CostLedger
        observed_at_start=clock()
        self.output, self.clock = Path(output), clock
        self.output.mkdir(parents=True, exist_ok=True)
        self.ledger = CostLedger(self.output/"t2_physical_ledger.jsonl")
        previous = self.output/"t2_costs.json"
        if previous.exists():
            prior = json.loads(previous.read_text())
            # Preserve the entire preceding public snapshot before overwrite.
            archive = self.output/"t2_cost_attempts"
            archive.mkdir(exist_ok=True)
            stamp = f"{time.time_ns()}_{os.getpid()}"
            self.prior_snapshot = str(archive/f"before_attempt_{stamp}.json")
            shutil.copy2(previous, self.prior_snapshot)
            if not self.ledger.attempts:
                # A completed legacy snapshot has only its current-attempt cost.
                # Reused caches imply that earlier work cannot be reconstructed.
                ambiguous = any(c.get("reused") for c in prior.get("checkpoints", []))
                self.ledger.begin_attempt(prior.get("slurm_job_id"), resume=ambiguous)
                self.ledger.add("gpu_wall_seconds", prior.get("current_total_seconds",0.))
                self.ledger.add("transformer_scored_tokens", prior.get("transformer_scored_tokens",0))
                self.ledger.add("lm_head_scored_positions",sum(c.get("lm_head_scored_positions",0)
                    for c in prior.get("checkpoints",[]) if not c.get("reused")))
                self.ledger.add("completed_attempts",1)
        elif not self.ledger.attempts:
            # A killed legacy invocation may have finished individual caches but
            # never published costs. Account only known completed checkpoint work.
            known=[]
            for path in sorted((self.output/"t2_logits").glob("*.json")):
                item=json.loads(path.read_text())
                if item.get("status")=="complete":
                    known.append(item)
            if known:
                self.ledger.begin_attempt(None,resume=True)
                self.ledger.add("gpu_wall_seconds",sum(c.get("elapsed_seconds",0.) for c in known))
                self.ledger.add("transformer_scored_tokens",sum(c.get("transformer_scored_tokens",0) for c in known))
                self.ledger.add("lm_head_scored_positions",sum(c.get("lm_head_scored_positions",0) for c in known))
                self.ledger.add("completed_attempts",1)
        unfinished=len(self.ledger.attempts)>self.ledger.get("completed_attempts",0)
        self.attempt=self.ledger.begin_attempt(os.environ.get("SLURM_JOB_ID"),resume=unfinished)
        self.last_observed=observed_at_start

    def observe(self):
        now=self.clock()
        self.ledger.add("gpu_wall_seconds",now-self.last_observed)
        self.last_observed=now

    def checkpoint(self, meta):
        if not meta.get("reused"):
            self.ledger.add("transformer_scored_tokens",meta.get("transformer_scored_tokens",0))
            self.ledger.add("lm_head_scored_positions",meta.get("lm_head_scored_positions",0))
            self.ledger.add("model_loading_seconds",meta.get("model_loading_seconds",0.))
            self.ledger.add("scoring_seconds",meta.get("scoring_seconds",0.))

    def finish(self):
        self.observe()
        self.ledger.add("completed_attempts",1)

    def fields(self):
        return dict(physical_ledger=str(self.ledger.path), physical_attempt_id=self.attempt["attempt_id"],
                    physical_gpu_hours=self.ledger.get("gpu_wall_seconds",0.)/3600,
                    physical_transformer_scored_tokens=self.ledger.get("transformer_scored_tokens",0),
                    physical_lm_head_scored_positions=self.ledger.get("lm_head_scored_positions",0),
                    physical_model_loading_seconds=self.ledger.get("model_loading_seconds",0.),
                    physical_scoring_seconds=self.ledger.get("scoring_seconds",0.),
                    physical_cost_completeness=self.ledger["counter_completeness"],
                    physical_attempts=self.ledger.attempts, physical_job_ids=self.ledger.job_ids,
                    interrupted_or_incomplete_attempts=len(self.ledger.attempts)-self.ledger.get("completed_attempts",0))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def checkpoint_stamp(path):
    """Small metadata only: no model-weight hashing or content scans."""
    path=Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint directory unavailable: {path}")
    files={}
    for pattern in ("config.json","tokenizer*.json","*.safetensors","*.bin","*.index.json"):
        for item in path.glob(pattern):
            info=item.stat()
            files[item.name]=dict(name=item.name,bytes=info.st_size,mtime_ns=info.st_mtime_ns)
    if not files:
        raise RuntimeError("checkpoint has no metadata/weight files")
    return dict(path=str(path.resolve()),files=[files[k] for k in sorted(files)])


def legacy_proof_matches(path, stamp, proof_path):
    """Legacy cache migration needs explicit immutable manifest metadata proof."""
    proof=json.loads(Path(proof_path).read_text())
    candidates=[proof.get("student_base",{})]+proof.get("route",[])
    for record in candidates:
        if not record.get("checkpoint") or Path(record["checkpoint"]).resolve()!=Path(path).resolve():
            continue
        files=record.get("metadata",{}).get("files",[])
        recorded={Path(f["path"]).name:dict(name=Path(f["path"]).name,bytes=f["size_bytes"],mtime_ns=f["mtime_ns"])
                  for f in files}
        return all(recorded.get(item["name"])==item for item in stamp["files"])
    return False


def validated_cache(path, cache_path, digest, legacy_cache_proof=None):
    meta_path=cache_path.with_suffix(".json")
    meta=json.loads(meta_path.read_text())
    if meta.get("status")!="complete" or meta.get("manifest_digest")!=digest or meta.get("checkpoint")!=str(path):
        raise RuntimeError("cache provenance mismatch; refusing silent cache reuse")
    current_stamp=checkpoint_stamp(path)
    if "checkpoint_stamp" not in meta:
        if legacy_cache_proof is None or not legacy_proof_matches(path,current_stamp,legacy_cache_proof):
            raise RuntimeError("legacy cache has no checkpoint stamp; explicit matching --legacy-cache-proof is required")
        meta["checkpoint_stamp"]=current_stamp
        meta["legacy_stamp_migration"]=dict(proof=str(legacy_cache_proof),
            proof_digest=hashlib.sha256(Path(legacy_cache_proof).read_bytes()).hexdigest(),
            weight_hash_computed=False,transformer_rerun=False,time_unix=time.time())
        write_json(meta_path,meta)
    if meta["checkpoint_stamp"]!=current_stamp:
        raise RuntimeError("checkpoint size/mtime metadata changed; refusing stale cache")
    cached=np.load(cache_path,mmap_mode="r")
    if list(cached.shape)!=meta["shape"] or cached.dtype!=np.float32:
        raise RuntimeError("cache shape/dtype mismatch")
    return meta


def load_records(path):
    text = Path(path).read_text()
    if Path(path).suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    result = json.loads(text)
    return result if isinstance(result, list) else result["records"]


def validate_records(records, max_questions=32, max_positions=16):
    """Return canonical records. Positions are zero-based completion offsets."""
    if not records:
        raise ValueError("empty fixed-prefix manifest")
    result, task_splits, keys = [], {}, set()
    for original in records:
        row = dict(original)
        row["rowid"] = str(row.get("rowid", row.get("row_id", row.get("id", ""))))
        row["split"] = {"fit": "calibration", "holdout": "heldout"}.get(row["split"], row["split"])
        if row["split"] not in {"calibration", "heldout"}:
            raise ValueError("only calibration/heldout diagnostic splits are allowed")
        if not row["rowid"] or not row.get("source"):
            raise ValueError("rowid/source provenance is required")
        if not row.get("checkpoint") and not row.get("provenance"):
            raise ValueError("origin checkpoint or explicit historical behavior provenance is required")
        row.setdefault("checkpoint", None)
        if "gold" in row["source"].lower():
            raise ValueError("gold prefixes are forbidden")
        key = (row["rowid"], row["source"])
        if key in keys:
            raise ValueError("one fixed answer per question/source is required")
        keys.add(key)
        if task_splits.setdefault(row["rowid"], row["split"]) != row["split"]:
            raise ValueError("same question leaks across coefficient fit and heldout")
        prompt, completion = row["prompt_ids"], row["completion_ids"]
        positions = row["positions"]
        if not prompt or not completion or not positions:
            raise ValueError("nonempty prompt/completion/positions required")
        if len(positions) > max_positions or len(set(positions)) != len(positions):
            raise ValueError("positions must be unique, maximum 16 per answer")
        if any(type(p) is not int or p < 0 or p >= len(completion) for p in positions):
            raise ValueError("positions must be valid completion prediction offsets")
        if any(type(t) is not int or t < 0 for t in prompt + completion):
            raise ValueError("token IDs must be nonnegative integers")
        row["positions"] = sorted(positions)
        row.setdefault("family", "unknown")
        result.append(row)
    if len(task_splits) > max_questions:
        raise ValueError("more than 32 diagnostic questions")
    if set(task_splits.values()) != {"calibration", "heldout"}:
        raise ValueError("both disjoint question splits are required")
    return result


def expand_records(records):
    return [dict(rowid=r["rowid"], source=r["source"], family=r["family"],
                 split=r["split"], position=p, checkpoint=r["checkpoint"])
            for r in records for p in r["positions"]]


def position_weights(rows, split, source=None):
    """Equal sources, equal questions within source, equal positions within answer.

    Missing source/question cells are not imputed. Their coverage is reported.
    """
    selected = [i for i, r in enumerate(rows) if r["split"] == split and (source is None or r["source"] == source)]
    weights = np.zeros(len(rows), dtype=np.float64)
    groups = defaultdict(lambda: defaultdict(list))
    for i in selected:
        groups[rows[i]["source"]][rows[i]["rowid"]].append(i)
    for tasks in groups.values():
        for indices in tasks.values():
            weights[indices] = 1. / len(groups) / len(tasks) / len(indices)
    return weights


def exact_kl(logits_q, logits_p):
    """Full global vocabulary normalization in FP32, FP64 reduction; no clamp."""
    logq = F.log_softmax(logits_q.float(), dim=-1)
    logp = F.log_softmax(logits_p.float(), dim=-1)
    return (logq.exp().double() * (logq.double() - logp.double())).sum(-1)


def endpoint_derivative(base, delta, q_logits, a, weights):
    """Analytic convex-objective derivative, one scalar shared across positions."""
    q = F.softmax(q_logits.float(), dim=-1)
    end = F.softmax(base.float() + float(a) * delta.float(), dim=-1)
    per_position = ((end.double() - q.double()) * delta.double()).sum(-1)
    return float((per_position * weights.to(per_position)).sum())


def fit_coefficient(base, delta, q_logits, weights, *, max_iterations=32, tolerance=1e-7):
    if weights.ndim != 1 or len(weights) != len(base) or float(weights.sum()) <= 0:
        raise ValueError("positive calibration position weights required")
    if abs(float(weights.sum()) - 1.) > 1e-6 or bool((weights < 0).any()):
        raise ValueError("calibration weights must sum to one")
    low_d = endpoint_derivative(base, delta, q_logits, 0., weights)
    high_d = endpoint_derivative(base, delta, q_logits, 1., weights)
    if low_d >= 0:
        return {"a": 0., "iterations": 0, "derivative": low_d, "boundary": "lower"}
    if high_d <= 0:
        return {"a": 1., "iterations": 0, "derivative": high_d, "boundary": "upper"}
    low, high = 0., 1.
    for iteration in range(1, max_iterations + 1):
        value = (low + high) / 2
        derivative = endpoint_derivative(base, delta, q_logits, value, weights)
        if abs(derivative) < tolerance or high - low < tolerance:
            break
        if derivative > 0:
            high = value
        else:
            low = value
    return {"a": value, "iterations": iteration, "derivative": derivative, "boundary": "interior"}


def chunked_fit(base, delta, q_logits, weights, *, chunk=32, tolerance=1e-7):
    """GPU-friendly fitting: exact sums, bounded temporary full-vocab buffers."""
    expected_q = torch.empty(len(base), dtype=torch.float64, device=base.device)
    for start in range(0, len(base), chunk):
        stop = start + chunk
        expected_q[start:stop] = (F.softmax(q_logits[start:stop].float(), -1).double() * delta[start:stop].double()).sum(-1)
    def derivative(a):
        total = torch.zeros((), dtype=torch.float64, device=base.device)
        for start in range(0, len(base), chunk):
            stop = start + chunk
            end = F.softmax(base[start:stop] + float(a) * delta[start:stop], -1)
            expected = (end.double() * delta[start:stop].double()).sum(-1)
            total += ((expected - expected_q[start:stop]) * weights[start:stop]).sum()
        return float(total)
    low_d, high_d = derivative(0.), derivative(1.)
    if low_d >= 0:
        return dict(a=0., iterations=0, derivative=low_d, boundary="lower")
    if high_d <= 0:
        return dict(a=1., iterations=0, derivative=high_d, boundary="upper")
    low, high = 0., 1.
    for iteration in range(1, 33):
        value = (low + high) / 2
        grad = derivative(value)
        if abs(grad) < tolerance or high-low < tolerance:
            break
        if grad > 0:
            high = value
        else:
            low = value
    return dict(a=value, iterations=iteration, derivative=grad, boundary="interior")


def summarize_values(values, rows):
    values = np.asarray(values, dtype=np.float64)
    result = {}
    for source in ["all"] + sorted({r["source"] for r in rows}):
        for split in ("calibration", "heldout"):
            weights = position_weights(rows, split, None if source == "all" else source)
            chosen = weights > 0
            key = (source, split)
            result[key] = dict(kl=float(values @ weights) if chosen.any() else None,
                               n_questions=len({rows[i]["rowid"] for i in np.flatnonzero(chosen)}),
                               n_positions=int(chosen.sum()),
                               minimum_position_kl=float(values[chosen].min()) if chosen.any() else None,
                               negative_position_kl_count=int((values[chosen] < 0).sum()))
    return result


def coverage(records):
    total = {r["rowid"] for r in records}
    result = []
    for source in sorted({r["source"] for r in records}):
        selected = [r for r in records if r["source"] == source]
        ids = {r["rowid"] for r in selected}
        result.append(dict(source=source, questions=len(ids), possible_questions=len(total),
                           missing_question_ids=sorted(total-ids),
                           calibration_questions=sum(r["split"] == "calibration" for r in selected),
                           heldout_questions=sum(r["split"] == "heldout" for r in selected),
                           positions=sum(len(r["positions"]) for r in selected),
                           checkpoints=sorted({r["checkpoint"] for r in selected if r["checkpoint"]}),
                           exact_origin_checkpoint_missing=sum(not r["checkpoint"] for r in selected)))
    return result


@torch.inference_mode()
def score_checkpoint(path, records, cache_path, *, digest, device, label, legacy_cache_proof=None):
    from .mpc_runtime import load_hidden
    started = time.monotonic()
    meta_path = cache_path.with_suffix(".json")
    if meta_path.exists() and cache_path.exists():
        meta=validated_cache(path,cache_path,digest,legacy_cache_proof)
        return dict(meta,reused=True,current_elapsed_seconds=time.monotonic()-started)
    if not path.is_dir() or "qwen" in str(path).lower():
        raise FileNotFoundError(f"missing or banned checkpoint: {path}")
    frozen_stamp=checkpoint_stamp(path)
    torch.cuda.reset_peak_memory_stats(device)
    loading_started = time.monotonic()
    model = load_hidden(ROOT, path, device, trainable=False).eval().requires_grad_(False)
    loading_seconds = time.monotonic()-loading_started
    weight = model.lm_head.weight.detach().float()
    bias = None if model.lm_head.bias is None else model.lm_head.bias.detach().float()
    vocab = weight.shape[0]
    count = sum(len(r["positions"]) for r in records)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float32, shape=(count, vocab))
    scored_tokens, offset = 0, 0
    scoring_started = time.monotonic()
    try:
        for record in records:
            # Predict completion offset j using hidden state at prompt_len+j-1.
            tokens = record["prompt_ids"] + record["completion_ids"][:max(record["positions"])]
            ids = torch.tensor([tokens], dtype=torch.long, device=device)
            if int(ids.max()) >= vocab:
                raise ValueError("manifest token ID lies outside shared vocabulary")
            attention = torch.ones_like(ids)
            hidden = model(ids, attention)
            indices = torch.tensor([len(record["prompt_ids"])+j-1 for j in record["positions"]], device=device)
            with torch.autocast(device_type="cuda", enabled=False):
                logits = F.linear(hidden[0, indices].float(), weight, bias)
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("nonfinite checkpoint logits")
            cached[offset:offset+len(indices)] = logits.cpu().numpy()
            offset += len(indices)
            scored_tokens += len(tokens)
            del hidden, logits, ids, attention
        cached.flush()
        scoring_seconds = time.monotonic()-scoring_started
        if checkpoint_stamp(path)!=frozen_stamp:
            raise RuntimeError("checkpoint metadata changed during scoring")
        meta = dict(status="complete", label=label, checkpoint=str(path), manifest_digest=digest,
                    checkpoint_stamp=frozen_stamp,
                    shape=[count, vocab], dtype="float32", reused=False,
                    model_loading_seconds=loading_seconds, scoring_seconds=scoring_seconds,
                    transformer_scored_tokens=scored_tokens, lm_head_scored_positions=count,
                    cache_bytes=cache_path.stat().st_size, elapsed_seconds=time.monotonic()-started,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(meta_path, meta)
        return meta
    finally:
        del cached, weight, bias, model
        gc.collect()
        torch.cuda.empty_cache()


def model_paths(model_map):
    s0 = model_map["s0"]
    paths = {"S0": Path(s0 if isinstance(s0, str) else s0["path"])}
    route = model_map["teacher_route"]
    for i, record in enumerate(route):
        stage = i if isinstance(record, str) else int(record.get("stage", i))
        value = record if isinstance(record, str) else record.get("path", record.get("checkpoint"))
        paths[f"T{stage}"] = Path(value)
    return paths, int(model_map.get("M", max(int(k[1:]) for k in paths if k.startswith("T"))))


@torch.inference_mode()
def run(args):
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("remote command must first cd /blue/du.j/jinjiaguo/CAFD")
    started = time.monotonic()
    for path in (args.manifest, args.model_map, args.output_dir):
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError("all diagnostic files must stay in CAFD")
    legacy_cache_proof=getattr(args,"legacy_cache_proof",None)
    if legacy_cache_proof is not None and not Path(legacy_cache_proof).resolve().is_relative_to(ROOT):
        raise ValueError("legacy cache proof must stay in CAFD")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    physical = PhysicalAttemptLedger(output)
    records = validate_records(load_records(args.manifest))
    rows = expand_records(records)
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    paths, M = model_paths(json.loads(args.model_map.read_text()))
    if M <= 0:
        raise ValueError("Teacher route requires at least one transition")
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    cache_dir = output / "t2_logits"
    costs, failures, stage_results, per_question = [], [], [], []
    for label, path in paths.items():
        try:
            if args.skip_scoring:
                meta=validated_cache(path,cache_dir/f"{label}.npy",digest,legacy_cache_proof)
                costs.append(dict(meta, reused=True))
            else:
                costs.append(score_checkpoint(path, records, cache_dir/f"{label}.npy", digest=digest, device=device,
                                              label=label,legacy_cache_proof=legacy_cache_proof))
            physical.checkpoint(costs[-1])
            print(json.dumps({"event":"checkpoint_scored", "label":label, "cost":costs[-1]}), flush=True)
        except Exception as exc:
            failure = dict(stage="model_scoring", checkpoint=str(path), label=label, error=repr(exc), traceback=traceback.format_exc())
            failures.append(failure)
            write_json(output/f"t2_error_{label}.json", failure)
            gc.collect()
            torch.cuda.empty_cache()
        write_json(output/"t2_progress.json", dict(status="scoring", scored=[c["label"] for c in costs], errors=failures))
        physical.observe()
    available = {c["label"] for c in costs}
    math_started = time.monotonic()
    def tensor(label):
        return torch.tensor(np.load(cache_dir/f"{label}.npy", mmap_mode="r"), dtype=torch.float32, device=device)
    if {"S0", "T0", f"T{M}"}.issubset(available):
        base, first, last = tensor("S0"), tensor("T0"), tensor(f"T{M}")
        delta = last-first
        del last
        weights = torch.tensor(position_weights(rows, "calibration"), device=device)
        if not bool((weights > 0).any()):
            raise ValueError("no calibration positions")
        # Fit-only slices make heldout exclusion structural, not just zero weights.
        fit_indices = (weights > 0).nonzero(as_tuple=False).flatten()
        for stage in range(M+1):
            if f"T{stage}" not in available:
                failures.append(dict(stage=stage, error="missing checkpoint logits; not interpolated or fabricated"))
                continue
            try:
                if stage == 0:
                    q_logits = base.clone()
                elif stage == M:
                    q_logits = base+delta
                else:
                    middle = tensor(f"T{stage}")
                    q_logits = base+(middle-first)
                    del middle
                fixed_a = stage/M
                fit = chunked_fit(base[fit_indices], delta[fit_indices], q_logits[fit_indices], weights[fit_indices], chunk=args.position_chunk)
                arrays = {}
                for name, a in (("fixed", fixed_a), ("fitted", fit["a"])):
                    values = []
                    for start in range(0, len(rows), args.position_chunk):
                        stop = start+args.position_chunk
                        value = exact_kl(q_logits[start:stop], base[start:stop]+a*delta[start:stop])
                        if not bool(torch.isfinite(value).all()) or float(value.min()) < -1e-5:
                            raise RuntimeError("nonfinite or materially negative KL; refusing clamp")
                        values.extend(value.cpu().tolist())
                    arrays[name] = np.array(values)
                summaries = {name: summarize_values(value, rows) for name, value in arrays.items()}
                for source in ["all"]+sorted({r["source"] for r in records}):
                    row = dict(stage=stage, M=M, source=source, fixed_a=fixed_a, fitted_a=fit["a"],
                               fit_iterations=fit["iterations"], fit_derivative=fit["derivative"],
                               fit_boundary=fit["boundary"], fit_scope="one_scalar_all_sources_calibration_only")
                    for split in ("calibration", "heldout"):
                        for name in ("fixed", "fitted"):
                            item = summaries[name][(source, split)]
                            row[f"{name}_{split}_kl"] = item["kl"]
                            row[f"{name}_{split}_minimum_position_kl"] = item["minimum_position_kl"]
                            row[f"{name}_{split}_negative_position_kl_count"] = item["negative_position_kl_count"]
                        row[f"{split}_questions"] = summaries["fixed"][(source,split)]["n_questions"]
                        row[f"{split}_positions"] = summaries["fixed"][(source,split)]["n_positions"]
                    stage_results.append(row)
                for record in records:
                    selected = [i for i,r in enumerate(rows) if r["rowid"] == record["rowid"] and r["source"] == record["source"]]
                    per_question.append(dict(stage=stage, rowid=record["rowid"], source=record["source"], family=record["family"], split=record["split"],
                        checkpoint=record["checkpoint"], provenance=record.get("provenance"),
                        n_positions=len(selected), positions=record["positions"], fixed_a=fixed_a, fitted_a=fit["a"],
                        fixed_kl=float(arrays["fixed"][selected].mean()), fitted_kl=float(arrays["fitted"][selected].mean())))
                print(json.dumps({"event":"stage_complete", "result":[r for r in stage_results if r["stage"] == stage and r["source"] == "all"]}), flush=True)
                write_json(output/"t2_progress.json", dict(status="geometry", completed_stages=[r["stage"] for r in stage_results if r["source"] == "all"], errors=failures))
                del q_logits
            except Exception as exc:
                failures.append(dict(stage=stage, error=repr(exc), traceback=traceback.format_exc()))
                write_json(output/f"t2_error_stage_{stage}.json", failures[-1])
            physical.observe()
        del base, first, delta
    else:
        failures.append(dict(stage="geometry", error="S0, T0, or endpoint missing; no substitute checkpoints used"))
    shared = sorted([r for r in stage_results if r["source"] == "all"], key=lambda r:r["stage"])
    reversals = [dict(previous_stage=a["stage"], stage=b["stage"], previous_a=a["fitted_a"], fitted_a=b["fitted_a"])
                 for a,b in zip(shared,shared[1:]) if b["fitted_a"] < a["fitted_a"]-1e-6]
    if stage_results:
        with (output/"path_geometry.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(stage_results[0]))
            writer.writeheader()
            writer.writerows(stage_results)
    with (output/"path_geometry_per_question.jsonl").open("w") as handle:
        for row in per_question:
            handle.write(json.dumps(row)+"\n")
    result = dict(status="complete" if not failures else "partial", manifest=str(args.manifest), manifest_digest=digest,
                  coverage=coverage(records), weighting="equal available sources; equal questions within source; equal positions within answer",
                  fitted_parameter="one scalar per stage shared across all sources, fit on calibration question IDs only",
                  kl="KL(q_m || q_endpoint(a)); FP32 global vocabulary log_softmax, FP64 reduction, no clamp",
                  ordering_complete_route=len(shared) == M+1,
                  ordering_non_decreasing_with_tolerance_1e_6=not reversals, coefficient_reversals=reversals,
                  stage_results=stage_results, failures=failures, frozen_test_accessed=False,
                  limitations=["diagnostic fit-prefix sample, not independent generalization evidence",
                               "missing source/question cells are not imputed; inspect coverage",
                               "no residual threshold is assumed to establish path equivalence",
                               "target geometry is not a KD Student performance ceiling",
                               "fit-error comparisons require identical prefixes, target, KL direction and normalization"])
    write_json(output/"path_geometry.json", result)
    total = time.monotonic()-started
    physical.finish()
    write_json(output/"t2_costs.json", dict(status=result["status"], checkpoints=costs, failures=failures,
        current_total_seconds=total, geometry_seconds=time.monotonic()-math_started, current_gpu_hours=total/3600,
        transformer_scored_tokens=sum(c.get("transformer_scored_tokens",0) for c in costs if not c.get("reused")),
        cache_bytes=sum(c.get("cache_bytes",0) for c in costs),
        gpu_name=torch.cuda.get_device_name(device),
        peak_allocated_bytes=max([torch.cuda.max_memory_allocated(device)]+[c.get("peak_allocated_bytes",0) for c in costs]),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"), node=os.environ.get("HOSTNAME"),
        generated_tokens=0, optimizer_updates=0, verifier_calls=0, **physical.fields()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--position-chunk", type=int, default=32)
    parser.add_argument("--skip-scoring", action="store_true")
    parser.add_argument("--legacy-cache-proof",type=Path,
                        help="Explicit original frozen manifest for verified metadata-only legacy cache migration")
    args = parser.parse_args()
    if args.position_chunk < 1:
        parser.error("position chunk must be positive")
    result = run(args)
    print(json.dumps({"status":result["status"], "failures":result["failures"], "reversals":result["coefficient_reversals"]}), flush=True)


if __name__ == "__main__":
    main()
