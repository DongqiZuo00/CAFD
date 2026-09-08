"""Verified-Solution SFT capacity gate for the frozen Instruct Teacher."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
import yaml
from torch.nn.parallel import DistributedDataParallel

from .data import load_rows
from .modeling import TEACHER_ID, TEACHER_REVISION, assert_exact_tokenizer_pair, load_model, load_tokenizers
from .optimizer import FP32AdamW
from .train_capacity import _read_jsonl, _sft_update
from .training_common import (
    HiddenCausalLM,
    append_csv,
    barrier,
    evaluate,
    init_distributed,
    restore_rng,
    save_model_only,
    seed_everything,
    snapshot_rng,
    unwrap_hidden_model,
    wallclock,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cafd/manufactoria_teacher_capacity_v1.yaml"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_id = str(config["experiment"]["run_id"])
    allowed = {
        ("teacher_capacity_v1_chat_verified_sft", "teacher_capacity_v1"),
        ("mistral_cafd_disjoint_v7", "mistral_cafd_disjoint_v7"),
        ("mistral_cafd_only_v6", "mistral_cafd_only_v6"),
    }
    identity = (str(config["protocol"]), run_id)
    if identity not in allowed:
        raise RuntimeError(f"unexpected Teacher capacity identity: {identity}")
    capacity = config["teacher_capacity"]
    generation = config["generation"]
    output = root / "runs" / "cafd" / "experiments" / run_id / "teacher_capacity"
    artifact_root = root / "artifacts" / "cafd" / "experiments" / run_id
    state_root = root / "state" / "cafd" / "experiments" / run_id
    gate_path = output / "gate.json"
    if gate_path.exists():
        status = json.loads(gate_path.read_text(encoding="utf-8")).get("status")
        if status == "passed":
            return
        if status == "TEACHER_CAPACITY_BLOCKED":
            raise SystemExit(44)

    context = init_distributed()
    started = wallclock()
    seed_everything(int(config["seed"]), context.rank)
    cache = root / ".cache" / "huggingface" / "hub"
    teacher_tokenizer, student_tokenizer = load_tokenizers(cache)
    assert_exact_tokenizer_pair(teacher_tokenizer, student_tokenizer)
    development_rows = load_rows(root, "development")

    source_path = root / str(capacity["source_verified_solutions"])
    records = _read_jsonl(source_path)
    expected_records = int(capacity["expected_verified_solutions"])
    if len(records) != expected_records:
        raise RuntimeError(
            f"verified-solution count changed: expected={expected_records}, actual={len(records)}"
        )
    if any(not record.get("prompt_ids") or not record.get("completion_ids") for record in records):
        raise RuntimeError("verified-solution source contains an empty prompt or completion")

    teacher = load_model(
        TEACHER_ID,
        TEACHER_REVISION,
        cache_dir=cache,
        device=context.device,
        trainable=True,
    )
    assert_exact_tokenizer_pair(
        teacher_tokenizer,
        student_tokenizer,
        teacher_model=teacher,
    )
    wrapped = HiddenCausalLM(teacher)
    model = (
        DistributedDataParallel(
            wrapped,
            device_ids=[context.local_rank],
            broadcast_buffers=False,
        )
        if context.world_size > 1
        else wrapped
    )
    optimizer = FP32AdamW(model.parameters(), lr=float(capacity["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    output.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)
    resume_path = output / f"resume.rank{context.rank}.pt"
    start_update = 0
    if resume_path.exists():
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        unwrap_hidden_model(model).causal_lm.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        restore_rng(payload["rng"])
        start_update = int(payload["update"])

    curve = artifact_root / "teacher_capacity_curve.csv"
    fields = ["condition", "step", "correct", "total", "accuracy", "generated_tokens"]
    prompt_renderer = str(generation["prompt_renderer"])
    if start_update == 0:
        save_model_only(model, teacher_tokenizer, output / "TBase", context)
        score = evaluate(
            model,
            teacher_tokenizer,
            development_rows,
            max_prompt_length=2048,
            max_completion_length=int(generation["max_new_tokens"]),
            context=context,
            prompt_renderer=prompt_renderer,
        )
        if context.primary:
            append_csv(
                curve,
                {"condition": "Teacher Verified-Solution SFT", "step": 0, **score},
                fields,
            )

    rng = random.Random(int(config["seed"]))
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
        positions = _sft_update(
            model,
            optimizer,
            records,
            sequence[begin : begin + int(capacity["global_batch_size"])],
            teacher_tokenizer,
            context,
        )
        scheduler.step()
        step = update_index + 1
        if context.primary:
            write_json(
                state_root / "teacher_capacity.json",
                {
                    "stage": "teacher_verified_solution_sft",
                    "update": step,
                    "updates": int(capacity["updates"]),
                    "completion_positions": positions,
                },
            )
        if step % int(capacity["checkpoint_interval"]) == 0:
            torch.save(
                {
                    "model": unwrap_hidden_model(model).causal_lm.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng": snapshot_rng(),
                    "update": step,
                },
                resume_path,
            )
        if step % int(capacity["eval_interval"]) == 0 or step == int(capacity["updates"]):
            checkpoint = output / f"SFT{step}"
            save_model_only(model, teacher_tokenizer, checkpoint, context)
            score = evaluate(
                model,
                teacher_tokenizer,
                development_rows,
                max_prompt_length=2048,
                max_completion_length=int(generation["max_new_tokens"]),
                context=context,
                prompt_renderer=prompt_renderer,
            )
            if context.primary:
                append_csv(
                    curve,
                    {"condition": "Teacher Verified-Solution SFT", "step": step, **score},
                    fields,
                )
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
                if row["condition"] == "Teacher Verified-Solution SFT":
                    candidates.append(
                        (float(row["accuracy"]), int(row["step"]), int(row["correct"]))
                    )
        selected_accuracy, selected_step, selected_correct = max(
            candidates,
            key=lambda item: (item[0], -item[1]),
        )
    if context.primary:
        checkpoint = output / ("TBase" if selected_step == 0 else f"SFT{selected_step}")
        write_json(
            gate_path,
            {
                "status": "passed" if passed else "TEACHER_CAPACITY_BLOCKED",
                "run_id": run_id,
                "optimizer": "FP32AdamW-master-weights",
                "learning_rate": float(capacity["learning_rate"]),
                "development_accuracy": selected_accuracy,
                "development_correct": selected_correct,
                "selected_step": selected_step,
                "actual_updates": selected_step if passed else int(capacity["updates"]),
                "checkpoint": str(checkpoint.resolve()),
                "verified_solutions": len(records),
                "source_verified_solutions": str(source_path.resolve()),
                "generated_tokens": 0,
                "gpu_hours": (wallclock() - started) * context.world_size / 3600.0,
            },
        )
    barrier(context)
    if not passed:
        raise SystemExit(44)


if __name__ == "__main__":
    main()
