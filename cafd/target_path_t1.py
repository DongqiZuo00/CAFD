"""Independent full-categorical generation from a frozen linear-logit policy.

No training, teacher replay, top-k/top-p truncation, or synthetic EOS.  A
left-padded batch keeps one independent KV cache per distinct model.  Inactive
rows are masked (their extra padded computation is included in the ledger).
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

VERSION = "target-path-t1-v1"


def canonical_terms(terms):
    """Combine identities before loading models; q0 really needs only S0."""
    combined = defaultdict(float)
    for path, coefficient in (terms.items() if isinstance(terms, dict) else terms):
        value = float(coefficient)
        if not math.isfinite(value):
            raise ValueError("non-finite policy coefficient")
        p = Path(path)
        key = str(p.resolve()) if p.exists() else str(path)
        combined[key] += value
    result = {p: c for p, c in sorted(combined.items()) if c != 0.0}
    if not result:
        raise ValueError("empty policy")
    return result


def left_pad(prompts, pad_id, device):
    if not prompts or any(not p for p in prompts):
        raise ValueError("nonempty prompts required")
    width = max(map(len, prompts))
    ids = torch.full((len(prompts), width), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    for j, prompt in enumerate(prompts):
        ids[j, -len(prompt):] = torch.tensor(prompt, dtype=torch.long, device=device)
        attention[j, -len(prompt):] = 1
    return ids, attention


class CachedModel:
    """A frozen Transformer plus one persistent FP32 output head."""
    def __init__(self, model):
        self.model = model.eval().requires_grad_(False)
        self.backbone = model.model if hasattr(model, "model") else model.base_model
        head = model.get_output_embeddings()
        self.weight = head.weight.detach().float()
        self.bias = None if head.bias is None else head.bias.detach().float()

    @torch.inference_mode()
    def forward(self, ids, attention, cache=None):
        positions = (attention.long().cumsum(-1) - 1).clamp_min(0)[:, -ids.shape[1]:]
        # DynamicCache generates absolute cache_position from its actual length;
        # position_ids separately correct the per-row left padding offsets.
        output = self.backbone(input_ids=ids, attention_mask=attention,
            position_ids=positions, past_key_values=cache, use_cache=True, return_dict=True)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            logits = F.linear(output.last_hidden_state[:, -1].float(), self.weight, self.bias)
        return logits, output.past_key_values


class EnsemblePool:
    def __init__(self, root, device, tokenizer=None, loader=None):
        self.root, self.device = Path(root), torch.device(device)
        self.tokenizer, self.loader, self.models = tokenizer, loader, {}
        self.load_history = []

    def select(self, terms):
        terms = canonical_terms(terms)
        for path in list(self.models):
            if path not in terms:
                del self.models[path]
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        from .mistral_runtime import load_model
        loader = self.loader or load_model
        for path in terms:
            if path not in self.models:
                started = time.monotonic()
                model = loader(path, "", cache_dir=self.root / ".cache/huggingface/hub",
                               device=self.device, trainable=False)
                vocab = model.get_output_embeddings().weight.shape[0]
                config = model.config.get_text_config()
                if vocab != int(config.vocab_size) or model.get_input_embeddings().weight.shape[0] != vocab:
                    raise RuntimeError(f"loaded model vocabulary/head mismatch: {path}")
                if self.tokenizer is not None and vocab != len(self.tokenizer):
                    raise RuntimeError(f"loaded model/tokenizer vocabulary mismatch: {path}")
                self.models[path] = CachedModel(model)
                self.synchronize()
                self.load_history.append(dict(path=path, seconds=time.monotonic()-started,
                    fp32_head_bytes=self.models[path].weight.numel()*4,
                    verified_head_config_input_vocab_size=vocab))
        return terms

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def close(self):
        self.models.clear()
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def combined_step(models, terms, ids, attention, caches):
    logits = None
    for path, coefficient in terms.items():
        component, caches[path] = models[path].forward(ids, attention, caches.get(path))
        logits = coefficient * component if logits is None else logits + coefficient * component
    return logits


class GenerationFailure(RuntimeError):
    def __init__(self, message, partial):
        super().__init__(message)
        self.partial = partial


@torch.inference_mode()
def generate_batch(models, terms, tokenizer, prompts, seeds, *, device,
                   max_new_tokens=2048, sample=True, stopper=None):
    """Return one complete token trace per seed, plus actual forward counts."""
    from .mpc_runtime import CompletionStop
    if len(prompts) != len(seeds) or not max_new_tokens > 0:
        raise ValueError("prompt/seed mismatch or invalid generation length")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer has neither pad nor EOS")
    device = torch.device(device)
    ids, attention = left_pad(prompts, pad_id, device)
    sequence = ids.clone()
    prompt_width = ids.shape[1]
    stopper = stopper or CompletionStop(tokenizer, prompt_width)
    generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]
    active = torch.ones(len(prompts), dtype=torch.bool, device=device)
    traces = [dict(completion_ids=[], selected_log_probs=[], stop_reason=None) for _ in prompts]
    caches, cost = {}, dict(model_forward_calls=0, transformer_token_positions=0,
        useful_prefill_tokens=sum(map(len, prompts))*len(terms),
        output_head_positions=0, active_output_head_positions=0,
        model_scored_tokens=0, generation_seconds=0.0)
    started = time.monotonic()
    try:
        for step in range(max_new_tokens):
            logits = combined_step(models, terms, ids, attention, caches)
            cost["model_forward_calls"] += len(terms)
            cost["transformer_token_positions"] += ids.numel()*len(terms)
            cost["output_head_positions"] += len(prompts)*len(terms)
            cost["active_output_head_positions"] += int(active.sum())*len(terms)
            cost["model_scored_tokens"] += len(prompts)*len(terms)
            if not torch.isfinite(logits[active]).all():
                raise FloatingPointError("non-finite full-vocabulary policy logits")
            log_probs = logits.log_softmax(-1)
            next_ids = torch.full((len(prompts), 1), pad_id, dtype=torch.long, device=device)
            for j in active.nonzero(as_tuple=False).flatten().tolist():
                token = (int(torch.multinomial(log_probs[j].exp(), 1, generator=generators[j]))
                         if sample else int(log_probs[j].argmax()))
                next_ids[j, 0] = token
                traces[j]["completion_ids"].append(token)
                traces[j]["selected_log_probs"].append(float(log_probs[j, token]))
            sequence = torch.cat((sequence, next_ids), dim=1)
            matched = stopper(sequence, logits)
            for j in (matched & active).nonzero(as_tuple=False).flatten().tolist():
                traces[j]["stop_reason"] = ("eos" if int(next_ids[j, 0]) == tokenizer.eos_token_id
                                             else "closing_fence")
            active = active & ~matched
            if not active.any():
                break
            # Only generated tokens from still-active rows affect a future step.
            attention = torch.cat((attention, active.long()[:, None]), dim=1)
            ids = next_ids
        for trace in traces:
            if trace["stop_reason"] is None:
                trace["stop_reason"] = "max_new_tokens"
    except Exception as exc:
        cost["generation_seconds"] = time.monotonic()-started
        cost["generated_tokens"] = sum(len(t["completion_ids"]) for t in traces)
        for trace in traces:
            trace["error"] = f"{type(exc).__name__}: {exc}"
            trace["stop_reason"] = trace["stop_reason"] or "generation_error"
        raise GenerationFailure(str(exc), (traces, cost)) from exc
    finally:
        caches.clear()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    cost["generation_seconds"] = time.monotonic()-started
    cost["generated_tokens"] = sum(len(t["completion_ids"]) for t in traces)
    return traces, cost


def _write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n")
    os.replace(temporary, path)


def _checkpoint_identity(path):
    p = Path(path)
    if not p.exists():
        return dict(path=path, exists=False)
    files = []
    for pattern in ("config.json", "*.safetensors", "*.bin", "*.index.json"):
        for f in sorted(p.glob(pattern)):
            stat = f.stat()
            files.append(dict(name=f.name, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    return dict(path=str(p.resolve()), exists=True, files=files)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def verify_record(row, trace, tokenizer):
    from .verifier import score_completion_details, extract_contract_program, verify_program
    started = time.monotonic()
    output = tokenizer.decode(trace["completion_ids"], skip_special_tokens=True)
    record = dict(trace, output=output, generated_tokens=len(trace["completion_ids"]),
                  truncated=trace["stop_reason"] == "max_new_tokens", verifier_calls=0)
    try:
        program = extract_contract_program(output)
        if program is not None:
            record["verifier_calls"] += 1  # score_completion_details internally calls official verifier.
        details = score_completion_details(output, row["ground_truth"], "hierarchical")
        if program is not None:
            record["verifier_calls"] += 1  # Preserve full per-test results in addition to reward details.
        official = verify_program(program, row["ground_truth"]) if program is not None else {
            "valid": False, "all_passed": False, "pass_rate": 0., "results": [], "error": "invalid_format"}
        # Generation/infrastructure failures remain in the denominator and are not successes.
        record.update(reward=0. if trace.get("error") else details["reward"],
            full_pass=bool(not trace.get("error") and details["reward"] == 1.0),
            format_error=details["tier"] == "invalid_format", parse_error=details["tier"] == "format_only",
            tier=details["tier"], verifier=official, reward_details=details)
    except Exception as exc:
        record.update(reward=0., full_pass=False, format_error=False, parse_error=False,
            tier="verifier_error", verifier={"error": f"{type(exc).__name__}: {exc}"},
            error=record.get("error") or f"verifier: {type(exc).__name__}: {exc}")
    record["verifier_seconds"] = time.monotonic()-started
    return record


def run_policy(root, run_dir, policy_id, terms, rows, decoding, seed_table, device,
               pool=None, tokenizer=None, max_new_records=None):
    """Resume strictly identical per-policy records; failed items stay explicit.

    seed_table maps row ID to the ordered replicate seeds.  A greedy policy uses
    just its first seed.  No large model-weight hashing is performed.
    """
    from .data import bounded_prompt
    from .prompting import prompt_token_ids, PROMPT_RENDERER_MISTRAL3_INSTRUCT
    from .mistral_runtime import load_tokenizers
    root, run_dir = Path(root), Path(run_dir)
    if "/" in policy_id or ".." in policy_id:
        raise ValueError("unsafe policy id")
    policy_dir = run_dir / "policies" / policy_id
    policy_dir.mkdir(parents=True, exist_ok=True)
    terms = canonical_terms(terms)
    pool = pool or EnsemblePool(root, device, tokenizer)
    tokenizer = tokenizer or pool.tokenizer
    if tokenizer is None:
        _, tokenizer = load_tokenizers(root / ".cache/huggingface/hub")
    pool.tokenizer = tokenizer
    if decoding.get("temperature", 1.) != 1. or decoding.get("top_p", 1.) != 1. or decoding.get("top_k", 0) != 0:
        raise ValueError("T1 requires temperature=1, top_p=1, top_k=0")
    sample = bool(decoding.get("sample", decoding.get("do_sample", True)))
    count, maximum, batch_size = (int(decoding.get("count", 2 if sample else 1)),
                                 int(decoding.get("max_new_tokens", 2048)), int(decoding.get("batch_size", 8)))
    if not sample and count != 1:
        raise ValueError("greedy policy requires exactly one replicate")
    prepared = []
    for row in rows:
        row_id = str(row["id"])
        prompt = prompt_token_ids(tokenizer, bounded_prompt(row), renderer=PROMPT_RENDERER_MISTRAL3_INSTRUCT)
        for replicate in range(count):
            prepared.append((row, prompt, replicate, int(seed_table[row_id][replicate])))
    identity = dict(version=VERSION, policy_id=policy_id, terms=terms,
        checkpoints=[_checkpoint_identity(p) for p in terms], decoding=decoding,
        rows_digest=_digest(rows), prepared_digest=_digest([(str(r["id"]), p, j, s) for r,p,j,s in prepared]),
        tokenizer=dict(name=str(getattr(tokenizer,"name_or_path","")), vocab_size=len(tokenizer),
                       eos=tokenizer.eos_token_id, pad=tokenizer.pad_token_id),
        verifier_digest=hashlib.sha256((root/"cafd/verifier.py").read_bytes()).hexdigest())
    identity_path = policy_dir / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("policy resume identity mismatch; refusing mixed evaluation")
    _write_json(identity_path, identity)
    records_path, costs_path = policy_dir/"rollouts.jsonl", policy_dir/"costs.json"
    prior = [json.loads(line) for line in records_path.read_text().splitlines()] if records_path.exists() else []
    keys = [(str(r["row_id"]), r["replicate"]) for r in prior]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate persisted evaluation rows")
    expected = {(str(row["id"]),replicate) for row,_,replicate,_ in prepared}
    if not set(keys).issubset(expected):
        raise RuntimeError("persisted rows outside frozen evaluation")
    pending = [item for item in prepared if (str(item[0]["id"]),item[2]) not in set(keys)]
    if max_new_records is not None:
        pending = pending[:int(max_new_records)]
    costs = json.loads(costs_path.read_text()) if costs_path.exists() else dict(policy_id=policy_id,
        attempts=[], generation_seconds=0., verifier_seconds=0., model_load_seconds=0.,
        generated_tokens=0, model_forward_calls=0, transformer_token_positions=0,
        useful_prefill_tokens=0, output_head_positions=0, active_output_head_positions=0,
        model_scored_tokens=0, gpu_wall_seconds=0., total_wall_seconds=0.)
    if not pending:
        return dict(policy_id=policy_id, records_path=str(records_path), expected_records=len(prepared),
                    completed_records=len(prior), resumed_records=len(prior), costs=costs)
    started, load_offset = time.monotonic(), len(pool.load_history)
    if pool.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(pool.device)
    load_error = None
    try:
        pool.select(terms)
    except Exception as exc:
        load_error = f"{type(exc).__name__}: {exc}"
        costs["attempts"].append(dict(stage="load", error=load_error, traceback=traceback.format_exc()))
    new_loads = pool.load_history[load_offset:]
    costs["model_load_seconds"] += sum(x["seconds"] for x in new_loads)
    costs.setdefault("loads", []).extend(new_loads)
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset+batch_size]
        error = load_error
        if any(len(p) > int(decoding.get("max_prompt_tokens",4096)) for _,p,_,_ in batch):
            error = "prompt cap exceeded; refusing silent truncation"
        batch_cost = {}
        if error is None:
            try:
                traces, batch_cost = generate_batch(pool.models, terms, tokenizer,
                    [p for _,p,_,_ in batch], [s for _,_,_,s in batch], device=device,
                    max_new_tokens=maximum, sample=sample)
            except GenerationFailure as exc:
                traces, batch_cost = exc.partial
                error = f"{type(exc.__cause__).__name__}: {exc.__cause__}"
                costs["attempts"].append(dict(stage="generation", offset=offset, error=error,
                                               traceback=traceback.format_exc()))
        else:
            traces = [dict(completion_ids=[],selected_log_probs=[],stop_reason="infrastructure_error",error=error) for _ in batch]
        for key, value in batch_cost.items():
            costs[key] = costs.get(key,0)+value
        new_records = []
        for (row,prompt,replicate,seed),trace in zip(batch,traces):
            record = verify_record(row,trace,tokenizer)
            record.update(policy_id=policy_id, row_id=str(row["id"]), family=row["problem_family"],
                replicate=replicate, seed=seed, prompt_ids=prompt, sample=sample,
                sample_index=replicate, mode="sample" if sample else "greedy",
                completion_text=record["output"], status="error" if record.get("error") else "ok",
                error=record.get("error"), policy_terms=terms, identity_digest=_digest(identity))
            costs["verifier_seconds"] += record["verifier_seconds"]
            costs["verifier_calls"] = costs.get("verifier_calls",0)+record["verifier_calls"]
            new_records.append(record)
        with records_path.open("a", encoding="utf-8") as handle:
            for record in new_records:
                handle.write(json.dumps(record,ensure_ascii=False)+"\n")
            handle.flush()
            os.fsync(handle.fileno())
        elapsed = time.monotonic()-started
        costs["last_attempt_elapsed_seconds"] = elapsed
        costs["completed_records"] = len(prior)+min(offset+batch_size,len(pending))
        _write_json(costs_path,costs)
        print(json.dumps(dict(event="t1_batch",policy_id=policy_id,completed=costs["completed_records"],
                              expected=len(prepared),elapsed_seconds=elapsed,error=error)),flush=True)
    elapsed = time.monotonic()-started
    costs["total_wall_seconds"] += elapsed
    costs["gpu_wall_seconds"] += elapsed if pool.device.type == "cuda" else 0.
    if pool.device.type == "cuda":
        costs["peak_allocated_bytes"] = max(costs.get("peak_allocated_bytes",0),torch.cuda.max_memory_allocated(pool.device))
        costs["peak_reserved_bytes"] = max(costs.get("peak_reserved_bytes",0),torch.cuda.max_memory_reserved(pool.device))
    _write_json(costs_path,costs)
    return dict(policy_id=policy_id,records_path=str(records_path),expected_records=len(prepared),
                completed_records=len(prior)+len(pending),resumed_records=len(prior),costs=costs)
