"""Compute-bounded four-B200 Teacher route with a seven-day hard gate."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import torch
import yaml
from torch.nn.parallel import DistributedDataParallel

from .configuration import load_experiment_config
from .data import load_rows, prompt_stream
from .layout import experiment_layout
from .modeling import TEACHER_ID, TEACHER_REVISION, assert_exact_tokenizer_pair, load_model, load_tokenizers
from .optimizer import FP32AdamW
from .training_common import HiddenCausalLM, append_csv, barrier, evaluate, init_distributed, restore_rng, rlvr_update, save_model_only, seed_everything, snapshot_rng, unwrap_hidden_model, wallclock, write_json


def _consecutive_successes(curve_path: Path, through_step: int, required_correct: int) -> int:
    if not curve_path.exists():
        return 0
    records = []
    with curve_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] == "Teacher" and int(row["step"]) <= through_step:
                records.append((int(row["step"]), int(row["correct"])))
    count = 0
    for _, correct in sorted(records):
        count = count + 1 if correct >= required_correct else 0
    return count


def _select_six(candidate_root: Path, final_step: int) -> list[int]:
    available = sorted(int(path.name[1:]) for path in candidate_root.glob("T*") if path.is_dir() and path.name[1:].isdigit())
    if len(available) < 6:
        raise RuntimeError(f"Teacher route has only {len(available)} checkpoints")
    chosen: list[int] = []
    for target in [round(final_step * index / 5) for index in range(6)]:
        options = [step for step in available if step not in chosen]
        chosen.append(min(options, key=lambda step: (abs(step - target), step)))
    chosen.sort()
    if chosen[0] != 0 or chosen[-1] != final_step:
        raise RuntimeError(f"route endpoints were not preserved: {chosen}")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = load_experiment_config(root)
    layout = experiment_layout(root, config)
    existing_gate = layout.teacher / "gate.json"
    if existing_gate.exists():
        status = json.loads(existing_gate.read_text(encoding="utf-8")).get("status")
        if status == "passed" and (existing_gate.with_name("route.json")).exists():
            return
        if status == "TEACHER_ROUTE_FAILED":
            raise SystemExit(42)
    route = config["teacher_route"]
    generation = config["generation"]
    prompt_renderer = str(generation["prompt_renderer"])
    context = init_distributed()
    seed_everything(config["seed"], context.rank)
    cache = root / ".cache" / "huggingface" / "hub"
    teacher_tokenizer, student_tokenizer = load_tokenizers(cache)
    initial_checkpoint = route.get("initial_checkpoint")
    initial_identifier = str((root / initial_checkpoint).resolve()) if initial_checkpoint else TEACHER_ID
    initial_revision = "" if initial_checkpoint else TEACHER_REVISION
    causal_lm = load_model(initial_identifier, initial_revision, cache_dir=cache, device=context.device, trainable=True)
    assert_exact_tokenizer_pair(teacher_tokenizer, student_tokenizer, teacher_model=causal_lm)
    wrapped = HiddenCausalLM(causal_lm)
    model = DistributedDataParallel(wrapped, device_ids=[context.local_rank], broadcast_buffers=False) if context.world_size > 1 else wrapped
    optimizer = FP32AdamW(model.parameters(), lr=float(route["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    output = layout.teacher
    candidates = output / "candidates"
    candidates.mkdir(parents=True, exist_ok=True)
    resume_path = output / f"resume.rank{context.rank}.pt"
    start_update = 0
    generated_tokens = 0
    zero_variance_streak = 0
    first_started = wallclock()
    if resume_path.exists():
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        unwrap_hidden_model(model).causal_lm.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        restore_rng(payload["rng"])
        start_update = int(payload["update"])
        generated_tokens = int(payload.get("generated_tokens", 0))
        zero_variance_streak = int(payload.get("zero_variance_streak", 0))
        first_started = float(payload.get("first_started", first_started))

    train_rows = load_rows(root, "train")
    development_rows = load_rows(root, "development")
    stream = prompt_stream(train_rows, int(route["updates"]), int(route["prompts_per_update"]), config["seed"])
    curve_path = layout.artifact_root / "teacher_curve.csv"
    fields = ["condition", "step", "correct", "total", "accuracy", "generated_tokens"]
    if start_update == 0:
        save_model_only(model, teacher_tokenizer, candidates / "T0", context)
        score = evaluate(model, teacher_tokenizer, development_rows, max_prompt_length=int(route["max_prompt_length"]), max_completion_length=int(generation["max_new_tokens"]), context=context, prompt_renderer=prompt_renderer)
        if context.primary:
            append_csv(curve_path, {"condition": "Teacher", "step": 0, **score}, fields)

    consecutive = _consecutive_successes(curve_path, start_update, int(route["success_correct"]))
    final_step = start_update
    success = consecutive >= int(route["success_consecutive"])
    deadline_hit = False
    signal_collapsed = False
    for update_index in range(start_update, int(route["updates"])):
        if wallclock() - first_started >= float(route["wall_time_seconds"]):
            deadline_hit = True
            break
        local_indices = stream[update_index][context.rank :: context.world_size]
        score_mode = str(route["reward_mode"])
        metrics = rlvr_update(
            model, optimizer, teacher_tokenizer, train_rows, local_indices,
            update=update_index,
            rollouts_per_prompt=int(route["rollouts_per_prompt"]),
            max_prompt_length=int(route["max_prompt_length"]),
            max_completion_length=int(generation["max_new_tokens"]),
            temperature=float(route["temperature"]),
            top_p=float(route["top_p"]),
            score_mode=score_mode,
            context=context,
            prompt_renderer=prompt_renderer,
            raw_rollout_path=output / "raw_rollouts" / f"rank{context.rank}.jsonl",
        )
        scheduler.step()
        step = update_index + 1
        final_step = step
        generated_tokens += int(metrics["generated_tokens"])
        zero_variance_streak = zero_variance_streak + 1 if int(metrics["variable_reward_groups"]) == 0 else 0
        signal_collapsed = zero_variance_streak >= int(route["max_consecutive_zero_variance_updates"])
        if context.primary:
            write_json(layout.state_root / "teacher.json", {"stage": "teacher_hierarchical_rlvr", "update": step, "updates": int(route["updates"]), "mean_reward": metrics["mean_reward"], "variable_reward_groups": metrics["variable_reward_groups"], "positive_reward_rollouts": metrics["positive_reward_rollouts"], "zero_variance_streak": zero_variance_streak, "generated_tokens": generated_tokens, "max_tokens_per_update": int(route["prompts_per_update"]) * int(route["rollouts_per_prompt"]) * int(generation["max_new_tokens"])})
        if step % int(route["checkpoint_interval"]) == 0:
            torch.save({"model": unwrap_hidden_model(model).causal_lm.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": snapshot_rng(), "update": step, "generated_tokens": generated_tokens, "zero_variance_streak": zero_variance_streak, "first_started": first_started}, resume_path)
            save_model_only(model, teacher_tokenizer, candidates / f"T{step}", context)
        if signal_collapsed:
            break
        if step % int(route["eval_interval"]) == 0:
            score = evaluate(model, teacher_tokenizer, development_rows, max_prompt_length=int(route["max_prompt_length"]), max_completion_length=int(generation["max_new_tokens"]), context=context, prompt_renderer=prompt_renderer)
            if context.primary:
                append_csv(curve_path, {"condition": "Teacher", "step": step, **score}, fields)
            consecutive = consecutive + 1 if int(score["correct"]) >= int(route["success_correct"]) else 0
            if (
                step >= int(route.get("minimum_updates", 0))
                and consecutive >= int(route["success_consecutive"])
            ):
                success = True
                if not (candidates / f"T{step}").is_dir():
                    save_model_only(model, teacher_tokenizer, candidates / f"T{step}", context)
                break

    barrier(context)
    if context.primary:
        if not success:
            write_json(output / "gate.json", {"status": "TEACHER_ROUTE_FAILED", "reason": "seven_day_wall_time" if deadline_hit else ("reward_signal_collapsed" if signal_collapsed else "development_capability_not_acquired"), "actual_updates": final_step, "generated_tokens": generated_tokens, "zero_variance_streak": zero_variance_streak, "gpu_hours": (wallclock() - first_started) * context.world_size / 3600.0})
        else:
            selected = _select_six(candidates, final_step)
            selected_set = set(selected)
            candidate_root = candidates.resolve()
            for path in candidates.glob("T*"):
                if path.is_dir() and int(path.name[1:]) not in selected_set:
                    resolved = path.resolve()
                    if resolved.parent != candidate_root:
                        raise RuntimeError(f"unsafe candidate cleanup target: {resolved}")
                    shutil.rmtree(resolved)
            route_records = [{"step": step, "checkpoint": str((candidates / f"T{step}").resolve())} for step in selected]
            write_json(output / "route.json", {"status": "frozen", "checkpoints": route_records, "final_step": final_step})
            final_score = None
            with curve_path.open("r", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if int(row["step"]) == final_step:
                        final_score = {"correct": int(row["correct"]), "accuracy": float(row["accuracy"])}
            if final_score is None:
                raise RuntimeError(f"missing Teacher development score at T{final_step}")
            write_json(output / "gate.json", {"status": "passed", "development_correct": final_score["correct"], "development_accuracy": final_score["accuracy"], "selected_step": final_step, "checkpoint": route_records[-1]["checkpoint"], "actual_updates": final_step, "generated_tokens": generated_tokens, "gpu_hours": (wallclock() - first_started) * context.world_size / 3600.0})
    barrier(context)
    if not success:
        raise SystemExit(42)


if __name__ == "__main__":
    main()
