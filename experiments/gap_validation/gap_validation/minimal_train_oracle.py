from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

from .minimal_runtime import elapsed_seconds, peak_gpu_memory_bytes
from .modeling import enforce_text_policy_trainable, load_full_policy
from .prompting import pair_messages
from .splits import load_jsonl


class AdaptiveStopCallback(TrainerCallback):
    def __init__(self, milestones: set[int], stop_after: int):
        self.milestones = milestones
        self.stop_after = stop_after

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step in self.milestones or step == self.stop_after:
            control.should_save = True
        if step >= self.stop_after:
            control.should_training_stop = True
        return control


def load_completion_dataset(path: Path) -> Dataset:
    rows: list[dict[str, Any]] = []
    for row in load_jsonl(path):
        rows.append({"prompt": pair_messages("math", row["prompt"]), "completion": row["completion"]})
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--budget-cap", type=int, default=800)
    parser.add_argument("--stop-after", type=int, required=True)
    parser.add_argument("--save-milestones", default="100,200,400,800")
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()

    started = time.monotonic()
    set_seed(42)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    model = load_full_policy(args.model_id, args.revision, attn_implementation="sdpa")
    enforce_text_policy_trainable(model, args.output_dir / "trainable_parameters.json")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    per_device_batch = 2
    denominator = per_device_batch * world_size
    if 32 % denominator:
        raise ValueError("world size does not preserve the 32-sequence Oracle batch")
    gradient_accumulation = 32 // denominator
    milestones = {int(value) for value in args.save_milestones.split(",") if value}
    config = SFTConfig(
        output_dir=str(args.output_dir),
        seed=42,
        data_seed=42,
        max_steps=args.budget_cap,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        optim="adamw_torch_fused",
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=gradient_accumulation,
        max_length=16384,
        completion_only_loss=True,
        packing=False,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        report_to="none",
        save_strategy="no",
        shuffle_dataset=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=load_completion_dataset(args.dataset),
        processing_class=tokenizer,
        callbacks=[AdaptiveStopCallback(milestones, args.stop_after)],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    actual_step = int(trainer.state.global_step)
    trainer.save_model(str(args.output_dir / f"model-step-{actual_step}"))
    trainer.save_state()
    wall = elapsed_seconds(started)
    peak = peak_gpu_memory_bytes()
    if trainer.is_world_process_zero():
        summary = {
            "method": "student_oracle_supervision",
            "model_id": args.model_id,
            "revision": args.revision,
            "learning_rate": args.learning_rate,
            "optimizer_updates": actual_step,
            "budget_cap": args.budget_cap,
            "global_sequence_batch_size": 32,
            "generated_tokens": 0,
            "teacher_scored_tokens": 0,
            "verifier_calls_during_training": 0,
            "world_size": world_size,
            "wall_time_seconds_this_segment": wall,
            "peak_gpu_memory_bytes": peak,
            "seed": 42,
        }
        (args.output_dir / f"run_summary_step_{actual_step}.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
