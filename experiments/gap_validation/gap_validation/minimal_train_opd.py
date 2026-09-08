from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback, set_seed
from trl import DistillationConfig, DistillationTrainer

from .minimal_runtime import completion_tokens_from_history, elapsed_seconds, peak_gpu_memory_bytes
from .modeling import enforce_text_policy_trainable, load_full_policy
from .prompting import pair_messages
from .splits import load_jsonl


class MilestoneSaveCallback(TrainerCallback):
    def __init__(self, milestones: set[int]):
        self.milestones = milestones

    def on_step_end(self, args, state, control, **kwargs):
        if int(state.global_step) in self.milestones:
            control.should_save = True
        return control


def expanded_on_policy_dataset(path: Path) -> Dataset:
    rows: list[dict[str, Any]] = []
    for row in load_jsonl(path):
        for rollout_slot in range(4):
            rows.append(
                {
                    "prompt": pair_messages("math", row["prompt"]),
                    "problem_id": row["id"],
                    "stream_position": row["stream_position"],
                    "rollout_slot": rollout_slot,
                }
            )
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--student-revision", required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-milestones", default="50,100,200")
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()

    started = time.monotonic()
    set_seed(42)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, revision=args.student_revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    student = load_full_policy(args.student_model, args.student_revision, attn_implementation="sdpa")
    enforce_text_policy_trainable(student, args.output_dir / "trainable_parameters.json")
    teacher = load_full_policy(str(args.teacher), revision=None, attn_implementation="sdpa")
    teacher.requires_grad_(False)
    teacher.eval()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    per_device_batch = 8
    denominator = per_device_batch * world_size
    if 32 % denominator:
        raise ValueError("world size does not preserve 32 on-policy rollout slots")
    gradient_accumulation = 32 // denominator
    config = DistillationConfig(
        output_dir=str(args.output_dir),
        seed=42,
        data_seed=42,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        optim="adamw_torch_fused",
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=gradient_accumulation,
        max_completion_length=8192,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        beta=0.0,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ds3_gather_for_generation=False,
        use_vllm=False,
        logging_steps=1,
        report_to="none",
        save_strategy="no",
        remove_unused_columns=False,
        shuffle_dataset=False,
    )
    trainer = DistillationTrainer(
        model=student,
        teacher_model=teacher,
        args=config,
        train_dataset=expanded_on_policy_dataset(args.dataset),
        processing_class=tokenizer,
        callbacks=[
            MilestoneSaveCallback({int(value) for value in args.save_milestones.split(",") if value})
        ],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir / "final_model"))
    trainer.save_state()
    wall = elapsed_seconds(started)
    peak = peak_gpu_memory_bytes()
    if trainer.is_world_process_zero():
        generated_tokens = completion_tokens_from_history(trainer.state.log_history)
        summary = {
            "method": "student_final_endpoint_opd",
            "objective": "exact_full_vocabulary_forward_kl",
            "kl_direction": "D_KL(stopgrad(p_T400)||p_S)",
            "completion_positions_only": True,
            "student_model": args.student_model,
            "student_revision": args.student_revision,
            "teacher": str(args.teacher),
            "learning_rate": args.learning_rate,
            "optimizer_updates": int(trainer.state.global_step),
            "prompts_per_update": 8,
            "responses_per_prompt": 4,
            "rollout_slots_per_update": 32,
            "generated_tokens": generated_tokens,
            "teacher_scored_tokens": generated_tokens,
            "verifier_calls": 0,
            "world_size": world_size,
            "wall_time_seconds": wall,
            "peak_gpu_memory_bytes": peak,
            "seed": 42,
            "reference_implementation": "trl.DistillationTrainer",
        }
        (args.output_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
