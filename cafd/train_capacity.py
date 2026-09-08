"""Capacity-first Verified-Solution SFT control for the fixed Student."""

from __future__ import annotations

import argparse
import csv
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from .configuration import load_experiment_config
from .data import load_rows
from .layout import experiment_layout
from .modeling import STUDENT_ID, STUDENT_REVISION, assert_exact_tokenizer_pair, load_model, load_tokenizers
from .optimizer import FP32AdamW
from .training_common import HiddenCausalLM, append_csv, barrier, evaluate, init_distributed, make_sequence, restore_rng, save_model_only, seed_everything, snapshot_rng, unwrap_hidden_model, wallclock, write_json


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sft_update(model, optimizer, records, indices, tokenizer, context) -> int:
    wrapper = unwrap_hidden_model(model)
    local_records = [records[index] for index in indices[context.rank :: context.world_size]]
    local_positions = sum(len(record["completion_ids"]) for record in local_records)
    positions = torch.tensor(local_positions, dtype=torch.long, device=context.device)
    if context.world_size > 1:
        dist.all_reduce(positions)
    global_positions = int(positions.item())
    if global_positions == 0:
        raise RuntimeError("Verified-Solution SFT batch has no completion tokens")
    optimizer.zero_grad(set_to_none=True)
    for record_index, record in enumerate(local_records):
        ids, attention, _, mask = make_sequence(record["prompt_ids"], record["completion_ids"], tokenizer, context.device)
        sync = model.no_sync() if isinstance(model, DistributedDataParallel) and record_index != len(local_records) - 1 else nullcontext()
        with sync:
            hidden = model(ids, attention)
            selected = hidden[:, :-1][mask]
            labels = ids[:, 1:][mask]
            logits = F.linear(selected, wrapper.lm_head.weight, getattr(wrapper.lm_head, "bias", None)).float()
            loss_sum = F.cross_entropy(logits, labels, reduction="sum")
            (loss_sum * context.world_size / global_positions).backward()
    torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
    optimizer.step()
    optimizer.assert_fp32_states()
    return global_positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = load_experiment_config(root)
    layout = experiment_layout(root, config)
    capacity = config["capacity"]
    run_id = layout.run_id
    output = layout.capacity
    existing_gate = output / "gate.json"
    if existing_gate.exists():
        status = json.loads(existing_gate.read_text(encoding="utf-8")).get("status")
        if status == "passed":
            return
        if status == "CAPACITY_BLOCKED":
            raise SystemExit(43)
    generation = config["generation"]
    prompt_renderer = str(generation["prompt_renderer"])
    context = init_distributed()
    started = wallclock()
    seed_everything(config["seed"], context.rank)
    cache = root / ".cache" / "huggingface" / "hub"
    teacher_tokenizer, student_tokenizer = load_tokenizers(cache)
    assert_exact_tokenizer_pair(teacher_tokenizer, student_tokenizer)
    development_rows = load_rows(root, "development")
    solutions_path = layout.capacity / "verified_train_solutions.jsonl"
    records = _read_jsonl(solutions_path)
    if not records:
        if context.primary:
            write_json(solutions_path.with_name("gate.json"), {"status": "CAPACITY_BLOCKED", "reason": "no canonical verified solutions"})
        raise SystemExit(43)

    student = load_model(STUDENT_ID, STUDENT_REVISION, cache_dir=cache, device=context.device, trainable=True)
    assert_exact_tokenizer_pair(teacher_tokenizer, student_tokenizer, student_model=student)
    wrapped = HiddenCausalLM(student)
    model = DistributedDataParallel(wrapped, device_ids=[context.local_rank], broadcast_buffers=False) if context.world_size > 1 else wrapped
    optimizer = FP32AdamW(model.parameters(), lr=float(capacity["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    output.mkdir(parents=True, exist_ok=True)
    resume_path = output / f"resume.rank{context.rank}.pt"
    start_update = 0
    if resume_path.exists():
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        unwrap_hidden_model(model).causal_lm.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        restore_rng(payload["rng"])
        start_update = int(payload["update"])

    curve = layout.artifact_root / "student_curves.csv"
    fields = ["condition", "step", "correct", "total", "accuracy", "generated_tokens"]
    if start_update == 0:
        save_model_only(model, student_tokenizer, layout.student_base, context)
        score = evaluate(model, student_tokenizer, development_rows, max_prompt_length=2048, max_completion_length=int(generation["max_new_tokens"]), context=context, prompt_renderer=prompt_renderer)
        if context.primary:
            append_csv(curve, {"condition": "Student Base", "step": 0, **score}, fields)
            append_csv(curve, {"condition": "Verified-Solution SFT", "step": 0, **score}, fields)

    rng = random.Random(config["seed"])
    sequence: list[int] = []
    required = int(capacity["updates"]) * int(capacity["global_batch_size"])
    while len(sequence) < required:
        epoch = list(range(len(records)))
        rng.shuffle(epoch)
        sequence.extend(epoch)

    passed = False
    selected_step = 0
    selected_accuracy = 0.0
    selected_correct = 0
    for update_index in range(start_update, int(capacity["updates"])):
        begin = update_index * int(capacity["global_batch_size"])
        positions = _sft_update(model, optimizer, records, sequence[begin : begin + int(capacity["global_batch_size"])], student_tokenizer, context)
        scheduler.step()
        step = update_index + 1
        if context.primary:
            write_json(layout.state_root / "capacity.json", {"stage": "verified_solution_sft", "run_id": run_id, "update": step, "updates": int(capacity["updates"]), "completion_positions": positions})
        if step % int(capacity["checkpoint_interval"]) == 0:
            torch.save({"model": unwrap_hidden_model(model).causal_lm.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": snapshot_rng(), "update": step}, resume_path)
        if step % int(capacity["eval_interval"]) == 0 or step == int(capacity["updates"]):
            checkpoint = output / f"SFT{step}"
            save_model_only(model, student_tokenizer, checkpoint, context)
            score = evaluate(model, student_tokenizer, development_rows, max_prompt_length=2048, max_completion_length=int(generation["max_new_tokens"]), context=context, prompt_renderer=prompt_renderer)
            if context.primary:
                append_csv(curve, {"condition": "Verified-Solution SFT", "step": step, **score}, fields)
            if int(score["correct"]) >= int(capacity["success_correct"]):
                passed = True
                selected_step = step
                selected_accuracy = float(score["accuracy"])
                selected_correct = int(score["correct"])
                break

    barrier(context)
    if not passed:
        candidates: list[tuple[float, int, int]] = []
        with curve.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["condition"] == "Verified-Solution SFT":
                    candidates.append((float(row["accuracy"]), int(row["step"]), int(row["correct"])))
        selected_accuracy, selected_step, selected_correct = max(candidates, key=lambda item: (item[0], -item[1]))
    if context.primary:
        checkpoint = layout.student_base if selected_step == 0 else output / f"SFT{selected_step}"
        write_json(output / "gate.json", {"status": "passed" if passed else "CAPACITY_BLOCKED", "run_id": run_id, "optimizer": "FP32AdamW-master-weights", "learning_rate": float(capacity["learning_rate"]), "development_accuracy": selected_accuracy, "development_correct": selected_correct, "selected_step": selected_step, "actual_updates": selected_step if passed else int(capacity["updates"]), "checkpoint": str(checkpoint.resolve()), "verified_solutions": len(records), "generated_tokens": 0, "gpu_hours": (wallclock() - started) * context.world_size / 3600.0})
    barrier(context)
    if not passed:
        raise SystemExit(43)


if __name__ == "__main__":
    main()
