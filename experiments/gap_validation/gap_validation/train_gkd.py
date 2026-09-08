from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback, set_seed
from trl import DistillationConfig, DistillationTrainer

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
        prompt = pair_messages("math", row["prompt"])
        for rollout_slot in range(4):
            rows.append(
                {
                    "prompt": prompt,
                    "problem_id": row["id"],
                    "stream_position": row["stream_position"],
                    "rollout_slot": rollout_slot,
                }
            )
    return Dataset.from_list(rows)


class ProgressiveDistillationTrainer(DistillationTrainer):
    def __init__(self, *args, teacher_schedule: dict[int, Path], **kwargs):
        self.teacher_schedule = dict(sorted(teacher_schedule.items()))
        self._active_teacher_step = min(self.teacher_schedule)
        super().__init__(*args, **kwargs)

    def _switch_teacher_if_needed(self) -> None:
        step = int(self.state.global_step)
        candidates = [boundary for boundary in self.teacher_schedule if boundary <= step]
        target_boundary = max(candidates)
        if target_boundary == self._active_teacher_step:
            return
        old_teacher = self.teacher_model
        self.teacher_model = None
        del old_teacher
        gc.collect()
        torch.cuda.empty_cache()
        teacher = load_full_policy(str(self.teacher_schedule[target_boundary]), revision=None)
        teacher.requires_grad_(False)
        teacher.eval()
        self.teacher_model = self.accelerator.prepare_model(teacher, evaluation_mode=True)
        self._active_teacher_step = target_boundary

    def training_step(self, model, inputs, num_items_in_batch=None):
        self._switch_teacher_if_needed()
        return super().training_step(model, inputs, num_items_in_batch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--student-revision", default="15852e8c16360a2fea060d615a32b45270f8a8fc")
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/frozen/math_training/student_prompt_stream.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--mode", choices=("endpoint", "progressive"), default="endpoint")
    parser.add_argument("--teacher-route-root", type=Path)
    parser.add_argument("--save-milestones", default="25,50,100,150,200")
    args = parser.parse_args()

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

    config = DistillationConfig(
        output_dir=str(args.output_dir),
        seed=42,
        data_seed=42,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        optim="adamw_torch_fused",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
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
    callbacks = [
        MilestoneSaveCallback({int(value) for value in args.save_milestones.split(",") if value})
    ]
    common = dict(
        model=student,
        teacher_model=teacher,
        args=config,
        train_dataset=expanded_on_policy_dataset(args.dataset),
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    if args.mode == "progressive":
        if args.teacher_route_root is None:
            raise ValueError("--teacher-route-root is required for progressive mode")
        schedule = {
            0: args.teacher_route_root / "checkpoint-50",
            40: args.teacher_route_root / "checkpoint-100",
            80: args.teacher_route_root / "checkpoint-200",
            120: args.teacher_route_root / "checkpoint-300",
            160: args.teacher_route_root / "final_model",
        }
        trainer = ProgressiveDistillationTrainer(**common, teacher_schedule=schedule)
    else:
        trainer = DistillationTrainer(**common)
    trainer.train()
    trainer.save_model(str(args.output_dir / "final_model"))
    trainer.save_state()
    summary = {
        "mode": args.mode,
        "objective": "exact_full_vocabulary_forward_kl",
        "reference_implementation": "trl.DistillationTrainer",
        "beta": 0.0,
        "temperature": 1.0,
        "prompts_per_update": 8,
        "responses_per_prompt": 4,
        "rollout_slots_per_update": 32,
        "optimizer_updates": int(trainer.state.global_step),
        "seed": 42,
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
