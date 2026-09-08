"""Formal Direct, Endpoint, Progressive, and CAFD-v1 Student runs."""

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
import yaml

from .configuration import load_experiment_config
from .data import bounded_prompt, load_rows, prompt_stream, rollout_seed
from .layout import ExperimentLayout, experiment_layout
from .exact_forward_kl import (
    assert_finite_or_dump,
    completion_prediction_mask,
    iter_endpoint_kl_blocks,
    iter_relative_kl_blocks,
)
from .modeling import assert_exact_tokenizer_pair, load_model, load_tokenizers
from .optimizer import FP32AdamW
from .phase_controller import PhaseController, PhaseState, parameter_sha256
from .training_common import (
    checkpoint_complete,
    DistributedContext,
    HiddenCausalLM,
    append_csv,
    evaluate,
    generate_group,
    init_distributed,
    restore_rng,
    rlvr_update,
    save_model_only,
    seed_everything,
    snapshot_rng,
    unwrap_hidden_model,
    wallclock,
    write_json,
)


METHOD_LABELS = {
    "direct": "Direct RLVR",
    "endpoint": "Endpoint OPD",
    "progressive": "Progressive GKD",
    "cafd": "CAFD-v1",
}


def _load_teacher(root: Path, checkpoint: str, device: torch.device) -> HiddenCausalLM:
    model = load_model(
        checkpoint,
        "",
        cache_dir=root / ".cache" / "huggingface" / "hub",
        device=device,
        trainable=False,
    )
    return HiddenCausalLM(model).eval().requires_grad_(False)


def _teacher_set(method: str, layout: ExperimentLayout, device: torch.device) -> dict[str, HiddenCausalLM]:
    if method == "direct":
        return {}
    route = json.loads((layout.teacher / "route.json").read_text(encoding="utf-8"))
    checkpoints = [item["checkpoint"] for item in route["checkpoints"]]
    if len(checkpoints) != 6:
        raise RuntimeError("the frozen Teacher acquisition route must contain exactly six checkpoints")
    indices = {"endpoint": [5], "progressive": [1, 2, 3, 4, 5], "cafd": [0, 1, 2, 3, 4, 5]}[method]
    return {f"R{index}": _load_teacher(layout.root, checkpoints[index], device) for index in indices}


def _generate_rollouts(model, tokenizer, rows, indices, update, settings, max_new_tokens, context, prompt_renderer):
    rollouts: list[dict[str, Any]] = []
    for prompt_slot, row_index in enumerate(indices):
        row = rows[row_index]
        prompt_ids, completions = generate_group(
            model,
            tokenizer,
            bounded_prompt(row),
            num_rollouts=int(settings["rollouts_per_prompt"]),
            max_prompt_length=2048,
            max_completion_length=int(max_new_tokens),
            temperature=float(settings["temperature"]),
            top_p=float(settings["top_p"]),
            seed=rollout_seed(update, prompt_slot, 0),
            sample=True,
            prompt_renderer=prompt_renderer,
        )
        for sample_slot, completion_ids in enumerate(completions):
            rollouts.append(
                {
                    "row_id": row["id"],
                    "prompt_ids": prompt_ids,
                    "completion_ids": completion_ids,
                    "prompt_slot": prompt_slot,
                    "sample_slot": sample_slot,
                }
            )
    return rollouts


def _persist_rollouts(path: Path, update: int, rollouts: list[dict[str, Any]]) -> None:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        for item in rollouts:
            handle.write(json.dumps({"update": update + 1, **item}, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _curve_milestone_count(path: Path, condition: str, step: int) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(
            1
            for row in csv.DictReader(handle)
            if row["condition"] == condition and int(row["step"]) == step
        )


def _raw_rollout_path(output: Path, name: str) -> Path:
    candidate = Path(name)
    if candidate.name != name or candidate.suffix != ".jsonl":
        raise ValueError("--rollout-log-name must be a .jsonl basename")
    return output / candidate


def _validated_prior_gpu_hours(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("--prior-gpu-hours must be finite and non-negative")
    return value


def _physical_rollout_tokens(output: Path) -> int:
    paths = sorted(output.glob("raw_rollouts*.jsonl"))
    if not paths:
        raise RuntimeError(f"no rollout logs found in {output}")
    total = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                completion = json.loads(line).get("completion_ids")
                if not isinstance(completion, list):
                    raise RuntimeError(f"bad completion at {path}:{line_number}")
                total += len(completion)
    return total


def _make_sequence_batch(rollouts, tokenizer, device):
    sequences = [item["prompt_ids"] + item["completion_ids"] for item in rollouts]
    if not sequences:
        raise RuntimeError("cannot build an empty distillation microbatch")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise RuntimeError("distillation tokenizer requires a pad token")
    max_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), max_length),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    prompt_lengths = torch.tensor(
        [len(item["prompt_ids"]) for item in rollouts], dtype=torch.long, device=device
    )
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[row, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[row, :length] = 1
    prediction_mask = completion_prediction_mask(
        input_ids,
        attention_mask,
        prompt_lengths,
        eos_token_id=tokenizer.eos_token_id,
    )
    return input_ids, attention_mask, prediction_mask


def _distillation_update(
    method: str,
    model: HiddenCausalLM,
    optimizer: FP32AdamW,
    tokenizer,
    rollouts: list[dict[str, Any]],
    teachers: dict[str, HiddenCausalLM],
    phase_ref: HiddenCausalLM | None,
    phase_index: int,
    token_block_size: int,
    output: Path,
) -> tuple[float, int, float]:
    positions = sum(len(item["completion_ids"]) for item in rollouts)
    if positions <= 0:
        raise RuntimeError("all Student distillation rollouts were empty")
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    route = ["R1", "R2", "R3", "R4", "R5"]
    previous_route = ["R0", "R1", "R2", "R3", "R4"]
    sequence_batch_size = 4
    for batch_start in range(0, len(rollouts), sequence_batch_size):
        batch = rollouts[batch_start : batch_start + sequence_batch_size]
        ids, attention, mask = _make_sequence_batch(
            batch, tokenizer, next(model.parameters()).device
        )
        student_hidden = model(ids, attention)[:, :-1]
        if method == "endpoint":
            target_model = teachers["R5"]
            with torch.no_grad():
                target_hidden = target_model(ids, attention)[:, :-1].detach()
            blocks = iter_endpoint_kl_blocks(
                student_hidden,
                target_hidden,
                model.lm_head,
                target_model.lm_head,
                mask,
                token_block_size=token_block_size,
            )
            scored_multiplier = 1
        elif method == "progressive":
            target_model = teachers[route[phase_index]]
            with torch.no_grad():
                target_hidden = target_model(ids, attention)[:, :-1].detach()
            blocks = iter_endpoint_kl_blocks(
                student_hidden,
                target_hidden,
                model.lm_head,
                target_model.lm_head,
                mask,
                token_block_size=token_block_size,
            )
            scored_multiplier = 1
        elif method == "cafd":
            if phase_ref is None:
                raise RuntimeError("CAFD phase reference is missing")
            previous_model = teachers[previous_route[phase_index]]
            next_model = teachers[route[phase_index]]
            with torch.no_grad():
                phase_hidden = phase_ref(ids, attention)[:, :-1].detach()
                previous_hidden = previous_model(ids, attention)[:, :-1].detach()
                next_hidden = next_model(ids, attention)[:, :-1].detach()
            blocks = iter_relative_kl_blocks(
                student_hidden,
                phase_hidden,
                next_hidden,
                previous_hidden,
                model.lm_head,
                phase_ref.lm_head,
                next_model.lm_head,
                previous_model.lm_head,
                mask,
                token_block_size=token_block_size,
                gamma=1.0,
                temperature=1.0,
            )
            scored_multiplier = 2
        else:
            raise ValueError(method)

        microbatch_loss = None
        for block in blocks:
            assert_finite_or_dump(
                {"loss": block.loss_sum},
                batch_payload={
                    "method": method,
                    "phase": phase_index,
                    "row_ids": [item["row_id"] for item in batch],
                    "prompt_ids": [item["prompt_ids"] for item in batch],
                    "completion_ids": [item["completion_ids"] for item in batch],
                    "block": [block.start, block.stop],
                },
                dump_path=str(output / "first_nonfinite_batch.pt"),
            )
            microbatch_loss = block.loss_sum if microbatch_loss is None else microbatch_loss + block.loss_sum
            total_loss += float(block.loss_sum.detach())
        if microbatch_loss is None:
            raise RuntimeError("distillation microbatch has no completion positions")
        (microbatch_loss / positions).backward()
        del student_hidden
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()
    optimizer.assert_fp32_states()
    return total_loss / positions, positions * scored_multiplier, gradient_norm


def _bootstrap_gate_payload(
    gate: dict[str, Any],
    *,
    step: int,
    score: dict[str, int | float],
    loss: float,
    gradient_norm: float,
) -> dict[str, Any]:
    total = int(score["total"])
    token_limit_rate = float(score["token_limit_hits"]) / max(total, 1)
    passed = (
        step == int(gate["update"])
        and int(score["contract_valid"]) >= int(gate["minimum_contract_valid"])
        and int(score["parse_or_better"]) >= int(gate["minimum_parse_or_better"])
        and int(score["eos_terminated"]) >= int(gate["minimum_eos_terminated"])
        and token_limit_rate <= float(gate["maximum_token_limit_rate"])
        and math.isfinite(loss)
        and loss >= float(gate["minimum_loss"])
        and math.isfinite(gradient_norm)
        and gradient_norm >= float(gate["minimum_gradient_norm"])
    )
    return {
        "status": "passed" if passed else "CAFD_BOOTSTRAP_GATE_FAILED",
        "step": step,
        "loss": loss,
        "gradient_norm": gradient_norm,
        "token_limit_rate": token_limit_rate,
        "criteria": gate,
        "development": score,
    }


def _save_generic_resume(path, model, optimizer, scheduler, update, generated_tokens, teacher_scored_tokens):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "model": model.causal_lm.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": snapshot_rng(),
            "update": update,
            "generated_tokens": generated_tokens,
            "teacher_scored_tokens": teacher_scored_tokens,
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", choices=sorted(METHOD_LABELS), required=True)
    parser.add_argument("--rollout-log-name", default="raw_rollouts.jsonl")
    parser.add_argument("--prior-gpu-hours", type=float, default=0.0)

    args = parser.parse_args()
    root = args.root.resolve()
    method = args.method
    config = load_experiment_config(root)
    layout = experiment_layout(root, config)
    if (layout.method(method) / "selected.json").exists():
        return
    settings = config["student"]
    generation = config["generation"]
    prompt_renderer = str(generation["prompt_renderer"])
    student_base = layout.student_base
    context = init_distributed()
    if context.world_size != 1:
        raise RuntimeError("each formal Student method must own exactly one B200")
    seed_everything(config["seed"])
    cache = root / ".cache" / "huggingface" / "hub"
    teacher_tokenizer, student_tokenizer = load_tokenizers(cache)
    student = load_model(
        str(student_base),
        "",
        cache_dir=cache,
        device=context.device,
        trainable=True,
    )
    assert_exact_tokenizer_pair(teacher_tokenizer, student_tokenizer, student_model=student)
    model = HiddenCausalLM(student)
    optimizer = FP32AdamW(model.parameters(), lr=float(settings["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    teachers = _teacher_set(method, layout, context.device)
    for teacher in teachers.values():
        assert_exact_tokenizer_pair(teacher_tokenizer, student_tokenizer, teacher_model=teacher.causal_lm)

    output = layout.method(method)
    output.mkdir(parents=True, exist_ok=True)
    raw_rollout_path = _raw_rollout_path(output, args.rollout_log_name)
    prior_gpu_hours = _validated_prior_gpu_hours(args.prior_gpu_hours)

    resume = output / "resume.pt"
    controller = PhaseController(output / "phase_references") if method == "cafd" else None
    phase_ref: HiddenCausalLM | None = None
    phase_state: PhaseState | None = None
    start_update = 0
    generated_tokens = 0
    teacher_scored_tokens = 0
    if resume.exists():
        if method == "cafd":
            assert controller is not None
            phase_state, phase_ref, extra = controller.load_resume(
                resume,
                student=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=context.device,
            )
            start_update = phase_state.global_update
            generated_tokens = int(extra.get("generated_tokens", 0))
            teacher_scored_tokens = int(extra.get("teacher_scored_tokens", 0))
        else:
            payload = torch.load(resume, map_location="cpu", weights_only=False)
            model.causal_lm.load_state_dict(payload["model"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            restore_rng(payload["rng"])
            start_update = int(payload["update"])
            generated_tokens = int(payload.get("generated_tokens", 0))
            teacher_scored_tokens = int(payload.get("teacher_scored_tokens", 0))
    if start_update == 0 and prior_gpu_hours != 0.0:
        raise RuntimeError("prior recovery GPU-hours require a nonzero resume")

    train_rows = load_rows(root, "train")
    development_rows = load_rows(root, "development")
    stream = prompt_stream(train_rows, int(settings["updates"]), int(settings["prompts_per_update"]), config["seed"])
    curve = layout.artifact_root / "student_curves.csv"
    fields = ["condition", "step", "correct", "total", "accuracy", "generated_tokens"]
    condition = METHOD_LABELS[method]
    milestones = {int(step) for step in settings["milestones"]}
    started = wallclock()
    if start_update > 0 and start_update in milestones:
        milestone_count = _curve_milestone_count(curve, condition, start_update)
        if milestone_count > 1:
            raise RuntimeError(
                f"development curve has {milestone_count} duplicate rows for "
                f"{condition} step {start_update}"
            )
        if milestone_count == 0:
            checkpoint = output / f"step{start_update}"
            if not checkpoint_complete(checkpoint):
                save_model_only(model, student_tokenizer, checkpoint, context)
            score = evaluate(
                model,
                student_tokenizer,
                development_rows,
                max_prompt_length=2048,
                max_completion_length=int(generation["max_new_tokens"]),
                context=context,
                prompt_renderer=prompt_renderer,
            )
            append_csv(curve, {"condition": condition, "step": start_update, **score}, fields)
    elif start_update == 0:
        score = evaluate(
            model,
            student_tokenizer,
            development_rows,
            max_prompt_length=2048,
            max_completion_length=int(generation["max_new_tokens"]),
            context=context,
            prompt_renderer=prompt_renderer,
        )
        append_csv(curve, {"condition": condition, "step": 0, **score}, fields)
        if method == "cafd":
            assert controller is not None
            phase_ref, phase_state = controller.begin_phase(model, phase_index=0, global_update=0, prompt_cursor=0)

    for update_index in range(start_update, int(settings["updates"])):
        phase_updates = int(config["cafd"]["phase_updates"])
        phase_index = update_index // phase_updates
        if method == "cafd":
            assert controller is not None
            if phase_state is None or phase_state.phase_index != phase_index:
                if phase_ref is not None:
                    del phase_ref
                    gc.collect()
                    torch.cuda.empty_cache()
                phase_ref, phase_state = controller.begin_phase(
                    model,
                    phase_index=phase_index,
                    global_update=update_index,
                    prompt_cursor=update_index * int(settings["prompts_per_update"]),
                )
        gradient_norm = 0.0
        if method == "direct":
            mode = str(settings["direct_reward_mode"])
            metrics = rlvr_update(
                model,
                optimizer,
                student_tokenizer,
                train_rows,
                stream[update_index],
                update=update_index,
                rollouts_per_prompt=int(settings["rollouts_per_prompt"]),
                max_prompt_length=2048,
                max_completion_length=int(generation["max_new_tokens"]),
                temperature=float(settings["temperature"]),
                top_p=float(settings["top_p"]),
                score_mode=mode,
                context=context,
                prompt_renderer=prompt_renderer,
                raw_rollout_path=raw_rollout_path,
            )
            loss = float(metrics["loss"])
            generated_tokens += int(metrics["generated_tokens"])
        else:
            rollouts = _generate_rollouts(model, student_tokenizer, train_rows, stream[update_index], update_index, settings, generation["max_new_tokens"], context, prompt_renderer)
            _persist_rollouts(raw_rollout_path, update_index, rollouts)
            generated_tokens += sum(len(item["completion_ids"]) for item in rollouts)
            loss, scored, gradient_norm = _distillation_update(
                method,
                model,
                optimizer,
                student_tokenizer,
                rollouts,
                teachers,
                phase_ref,
                phase_index,
                int(config["cafd"]["token_block_size"]),
                output,
            )
            teacher_scored_tokens += scored
        scheduler.step()
        step = update_index + 1
        if phase_state is not None:
            phase_state.phase_local_update = step - phase_index * phase_updates
            phase_state.global_update = step
            phase_state.prompt_cursor = step * int(settings["prompts_per_update"])
            if step % phase_updates == 0 and parameter_sha256(phase_ref) != phase_state.phase_reference_hash:
                raise RuntimeError("phase-reference parameters changed within a CAFD phase")
        write_json(
            layout.state_root / f"{method}.json",
            {
                "stage": METHOD_LABELS[method],
                "update": step,
                "updates": int(settings["updates"]),
                "phase": phase_index,
                "phase_local_update": step - phase_index * phase_updates,
                "loss": loss,
                "gradient_norm": gradient_norm,
                "generated_tokens": generated_tokens,
                "teacher_scored_tokens": teacher_scored_tokens,
            },
        )
        # Scientific checkpoints remain at every configured milestone. Full
        # optimizer resumes align with 40-update phases to avoid synchronous
        # multi-method writes dominating B200 wall time.
        if step % int(config["cafd"]["phase_updates"]) == 0 or step == int(settings["updates"]):
            if method == "cafd":
                assert controller is not None and phase_state is not None
                controller.save_resume(
                    resume,
                    student=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    state=phase_state,
                    extra={"generated_tokens": generated_tokens, "teacher_scored_tokens": teacher_scored_tokens},
                )
            else:
                _save_generic_resume(resume, model, optimizer, scheduler, step, generated_tokens, teacher_scored_tokens)
        if step in milestones:
            checkpoint = output / f"step{step}"
            save_model_only(model, student_tokenizer, checkpoint, context)
            score = evaluate(
                model,
                student_tokenizer,
                development_rows,
                max_prompt_length=2048,
                max_completion_length=int(generation["max_new_tokens"]),
                context=context,
                prompt_renderer=prompt_renderer,
                include_details=(
                    method == "cafd"
                    and "bootstrap_gate" in config["cafd"]
                    and step == int(config["cafd"]["bootstrap_gate"]["update"])
                ),
            )
            curve_score = {
                key: score[key]
                for key in ("correct", "total", "accuracy", "generated_tokens")
            }
            append_csv(curve, {"condition": condition, "step": step, **curve_score}, fields)
            if (
                method == "cafd"
                and "bootstrap_gate" in config["cafd"]
                and step == int(config["cafd"]["bootstrap_gate"]["update"])
            ):
                gate_payload = _bootstrap_gate_payload(
                    config["cafd"]["bootstrap_gate"],
                    step=step,
                    score=score,
                    loss=loss,
                    gradient_norm=gradient_norm,
                )
                write_json(output / "bootstrap_gate.json", gate_payload)
                if gate_payload["status"] != "passed":
                    raise RuntimeError("CAFD_BOOTSTRAP_GATE_FAILED")

    candidates: list[tuple[float, int]] = []
    with curve.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] == condition:
                candidates.append((float(row["accuracy"]), int(row["step"])))
    best_accuracy, best_step = max(candidates, key=lambda item: (item[0], -item[1]))
    checkpoint = student_base if best_step == 0 else output / f"step{best_step}"
    current_gpu_hours = (wallclock() - started) / 3600.0
    actual_generated_tokens = _physical_rollout_tokens(output)
    scored_multiplier = {"direct": 0, "endpoint": 1, "progressive": 1, "cafd": 2}[method]
    if teacher_scored_tokens != generated_tokens * scored_multiplier:
        raise RuntimeError(f"logical teacher-scored token mismatch for {method}")
    if actual_generated_tokens < generated_tokens:
        raise RuntimeError(f"physical generated tokens dropped below logical tokens for {method}")
    actual_teacher_scored_tokens = actual_generated_tokens * scored_multiplier
    discarded_generated_tokens = actual_generated_tokens - generated_tokens
    discarded_teacher_scored_tokens = (
        actual_teacher_scored_tokens - teacher_scored_tokens
    )
    selection: dict[str, Any] = {
        "condition": condition,
        "development_accuracy": best_accuracy,
        "selected_step": best_step,
        "checkpoint": str(checkpoint.resolve()),
        "generated_tokens": generated_tokens,
        "teacher_scored_tokens": teacher_scored_tokens,
        "logical_generated_tokens": generated_tokens,
        "logical_teacher_scored_tokens": teacher_scored_tokens,
        "actual_generated_tokens": actual_generated_tokens,
        "actual_teacher_scored_tokens": actual_teacher_scored_tokens,
        "gpu_hours": prior_gpu_hours + current_gpu_hours,
    }
    if start_update > 0:
        selection.update(
            {
                "prior_gpu_hours": prior_gpu_hours,
                "current_attempt_gpu_hours": current_gpu_hours,
                "resume_start_update": start_update,
                "rollout_log": str(raw_rollout_path.resolve()),
                "prior_discarded_generated_tokens":
                    discarded_generated_tokens,
                "prior_discarded_teacher_scored_tokens":
                    discarded_teacher_scored_tokens,
            }
        )
    write_json(output / "selected.json", selection)


if __name__ == "__main__":
    main()
