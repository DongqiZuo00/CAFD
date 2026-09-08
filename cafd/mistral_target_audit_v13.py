"""Development-only numerical audit of the CAFD relative target."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .exact_forward_kl import completion_prediction_mask
from .mistral_runtime import assert_exact_tokenizer_pair, load_model, load_tokenizers
from .training_common import HiddenCausalLM

SOURCE_RUN = "mistral_cafd_disjoint_v7"
CAFD_RUN = "mistral_cafd_gen_v10_mixed"
AUDIT_ID = "mistral_cafd_target_audit_v13"
PHASE_STEPS = [0, 40, 80, 120, 160]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_hidden(root: Path, checkpoint: Path, device: torch.device) -> HiddenCausalLM:
    causal_lm = load_model(
        str(checkpoint), "", cache_dir=root / ".cache/huggingface/hub",
        device=device, trainable=False,
    )
    return HiddenCausalLM(causal_lm).eval().requires_grad_(False)


def phase_rollouts(path: Path, phase: int, count: int) -> list[dict[str, Any]]:
    source = f"R{phase + 1}_next_teacher"
    selected: list[dict[str, Any]] = []
    seen_rows: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            row_id = str(item.get("row_id", ""))
            if (
                int(item.get("support_phase", -1)) == phase
                and str(item.get("rollout_source")) == source
                and row_id not in seen_rows
                and item.get("completion_ids")
            ):
                selected.append(item)
                seen_rows.add(row_id)
                if len(selected) == count:
                    break
    if len(selected) != count:
        raise RuntimeError(f"phase {phase}: found {len(selected)} usable {source} rollouts")
    return selected


def sample_positions(item, eos_token_id, device, maximum):
    prompt = [int(x) for x in item["prompt_ids"]]
    completion = [int(x) for x in item["completion_ids"]]
    ids = torch.tensor([prompt + completion], dtype=torch.long, device=device)
    attention = torch.ones_like(ids)
    lengths = torch.tensor([len(prompt)], dtype=torch.long, device=device)
    mask = completion_prediction_mask(ids, attention, lengths, eos_token_id=eos_token_id)
    valid = torch.where(mask[0])[0]
    if valid.numel() == 0:
        raise RuntimeError(f"rollout {item['row_id']} has no completion positions")
    if valid.numel() > maximum:
        offsets = torch.linspace(0, valid.numel() - 1, steps=maximum, device=device).round().long()
        valid = valid[offsets]
    return ids, attention, valid, ids[0, valid + 1]


@torch.inference_mode()
def log_probs(model, ids, attention, positions):
    hidden = model(ids, attention)[:, :-1, :][0, positions]
    head = model.lm_head
    weight = head.weight.detach().float()
    bias = None if head.bias is None else head.bias.detach().float()
    logits = F.linear(hidden.float(), weight, bias)
    result = F.log_softmax(logits, dim=-1)
    del hidden, weight, bias, logits
    return result


def extend(store, name, tensor):
    store.setdefault(name, []).extend(float(x) for x in tensor.detach().cpu().tolist())


@torch.inference_mode()
def measure(phase_reference, previous_teacher, next_teacher, items, eos_token_id, device, maximum):
    values: dict[str, list[float]] = {}
    row_ids = []
    for item in items:
        ids, attention, positions, labels = sample_positions(
            item, eos_token_id, device, maximum
        )
        log_ref = log_probs(phase_reference, ids, attention, positions)
        log_previous = log_probs(previous_teacher, ids, attention, positions)
        log_next = log_probs(next_teacher, ids, attention, positions)
        log_q = F.log_softmax(log_ref + log_next - log_previous, dim=-1)
        p_ref, p_previous, p_next, p_q = (
            log_ref.exp(), log_previous.exp(), log_next.exp(), log_q.exp()
        )

        extend(values, "kl_q_to_next", (p_q * (log_q - log_next)).sum(-1))
        extend(values, "kl_next_to_q", (p_next * (log_next - log_q)).sum(-1))
        extend(values, "kl_previous_to_phase_ref", (p_previous * (log_previous - log_ref)).sum(-1))
        extend(values, "kl_phase_ref_to_previous", (p_ref * (log_ref - log_previous)).sum(-1))
        extend(values, "kl_q_to_phase_ref", (p_q * (log_q - log_ref)).sum(-1))
        extend(values, "entropy_q", -(p_q * log_q).sum(-1))
        extend(values, "entropy_next", -(p_next * log_next).sum(-1))
        extend(values, "entropy_phase_ref", -(p_ref * log_ref).sum(-1))
        extend(values, "maximum_probability_q", p_q.max(-1).values)
        extend(values, "maximum_probability_next", p_next.max(-1).values)

        q_top = log_q.topk(10, dim=-1).indices
        next_top = log_next.topk(10, dim=-1).indices
        overlap = (q_top.unsqueeze(2) == next_top.unsqueeze(1)).any(2).float().mean(1)
        extend(values, "top10_overlap_q_next", overlap)
        extend(values, "q_mass_on_next_top10", p_q.gather(1, next_top).sum(1))
        extend(values, "top1_agreement_q_next", (q_top[:, 0] == next_top[:, 0]).float())

        teacher_delta = log_next - log_previous
        desired_delta = log_next - log_ref
        teacher_delta -= teacher_delta.mean(-1, keepdim=True)
        desired_delta -= desired_delta.mean(-1, keepdim=True)
        residual = log_ref - log_previous
        residual -= residual.mean(-1, keepdim=True)
        extend(values, "teacher_delta_desired_cosine", F.cosine_similarity(teacher_delta, desired_delta, dim=-1))
        extend(values, "teacher_delta_rms", teacher_delta.square().mean(-1).sqrt())
        extend(values, "student_teacher_residual_rms", residual.square().mean(-1).sqrt())

        rows = torch.arange(labels.numel(), device=device)
        extend(values, "next_token_nll_q", -log_q[rows, labels])
        extend(values, "next_token_nll_next", -log_next[rows, labels])
        extend(values, "next_token_nll_phase_ref", -log_ref[rows, labels])
        row_ids.append(str(item["row_id"]))
        del ids, attention, positions, labels, log_ref, log_previous, log_next
        del log_q, p_ref, p_previous, p_next, p_q, teacher_delta, desired_delta, residual
        torch.cuda.empty_cache()

    summary: dict[str, Any] = {"row_ids": row_ids, "positions": len(next(iter(values.values())))}
    for name, samples in sorted(values.items()):
        ordered = sorted(samples)
        n = len(ordered)
        summary[name] = {
            "mean": sum(ordered) / n,
            "p50": ordered[n // 2],
            "p90": ordered[min(n - 1, math.floor(0.9 * n))],
            "maximum": ordered[-1],
        }
    return summary


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    metrics = [
        "kl_q_to_next", "kl_previous_to_phase_ref", "top1_agreement_q_next",
        "top10_overlap_q_next", "teacher_delta_desired_cosine",
        "teacher_delta_rms", "student_teacher_residual_rms",
        "next_token_nll_q", "next_token_nll_next",
    ]
    fields = [
        "phase", "student_reference_step", "teacher_previous_step",
        "teacher_next_step", "positions", *(f"{name}_mean" for name in metrics),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {
                "phase": record["phase"],
                "student_reference_step": record["student_reference_step"],
                "teacher_previous_step": record["teacher_previous_step"],
                "teacher_next_step": record["teacher_next_step"],
                "positions": record["metrics"]["positions"],
            }
            row.update({f"{name}_mean": record["metrics"][name]["mean"] for name in metrics})
            writer.writerow(row)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sequences-per-phase", type=int, default=2)
    parser.add_argument("--positions-per-sequence", type=int, default=64)
    args = parser.parse_args()
    root = args.root.resolve()
    if root != Path("/blue/du.j/jinjiaguo/CAFD"):
        raise RuntimeError(f"audit must run in the CAFD root, got {root}")
    if not torch.cuda.is_available():
        raise RuntimeError("target audit requires one CUDA GPU")
    if args.sequences_per_phase <= 0 or args.positions_per_sequence <= 0:
        raise ValueError("sample sizes must be positive")
    device = torch.device("cuda:0")

    source = root / "runs/cafd/experiments" / SOURCE_RUN
    cafd = root / "runs/cafd/experiments" / CAFD_RUN
    route = json.loads((source / "teacher/route.json").read_text(encoding="utf-8"))
    if route.get("status") != "frozen" or len(route.get("checkpoints", [])) != 6:
        raise RuntimeError("expected the frozen six-checkpoint Mistral Teacher route")
    teacher_paths = [Path(x["checkpoint"]) for x in route["checkpoints"]]
    teacher_steps = [str(x["step"]) for x in route["checkpoints"]]
    student_paths = [
        source / "student_base/S0",
        *(cafd / "cafd" / f"step{step}" for step in PHASE_STEPS[1:]),
    ]
    rollout_path = cafd / "cafd/raw_rollouts.jsonl"
    missing = [str(x) for x in [*teacher_paths, *student_paths, rollout_path] if not x.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")

    teacher_tokenizer, student_tokenizer = load_tokenizers(root / ".cache/huggingface/hub")
    assert_exact_tokenizer_pair(teacher_tokenizer, student_tokenizer)
    output = root / "artifacts/cafd/experiments" / AUDIT_ID
    result_path = output / "target_audit.json"
    csv_path = output / "target_audit.csv"
    state_path = root / "state/cafd/experiments" / AUDIT_ID / "audit.json"
    records = []

    previous = load_hidden(root, teacher_paths[0], device)
    for phase in range(5):
        next_teacher = load_hidden(root, teacher_paths[phase + 1], device)
        phase_reference = load_hidden(root, student_paths[phase], device)
        items = phase_rollouts(rollout_path, phase, args.sequences_per_phase)
        metrics = measure(
            phase_reference, previous, next_teacher, items,
            student_tokenizer.eos_token_id, device, args.positions_per_sequence,
        )
        records.append({
            "phase": phase,
            "student_reference_step": PHASE_STEPS[phase],
            "student_reference_checkpoint": str(student_paths[phase]),
            "teacher_previous_step": teacher_steps[phase],
            "teacher_previous_checkpoint": str(teacher_paths[phase]),
            "teacher_next_step": teacher_steps[phase + 1],
            "teacher_next_checkpoint": str(teacher_paths[phase + 1]),
            "metrics": metrics,
        })
        payload = {
            "audit_id": AUDIT_ID, "status": "running",
            "completed_phases": len(records), "frozen_test_accessed": False,
            "records": records,
        }
        write_json(result_path, payload)
        write_csv(csv_path, records)
        write_json(state_path, {
            "stage": "target_audit", "phase": phase,
            "completed_phases": len(records),
            "gpu": torch.cuda.get_device_name(device),
            "maximum_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
            "frozen_test_accessed": False,
        })
        del phase_reference, previous
        gc.collect()
        torch.cuda.empty_cache()
        previous = next_teacher

    del previous
    gc.collect()
    torch.cuda.empty_cache()
    payload = {
        "audit_id": AUDIT_ID, "status": "complete",
        "sample": {
            "sequences_per_phase": args.sequences_per_phase,
            "positions_per_sequence": args.positions_per_sequence,
            "rollout_source": "persisted_next_teacher_rollouts",
        },
        "frozen_test_accessed": False, "records": records,
    }
    write_json(result_path, payload)
    write_csv(csv_path, records)
    write_json(state_path, {
        "stage": "complete", "completed_phases": 5,
        "result": str(result_path), "frozen_test_accessed": False,
    })
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
