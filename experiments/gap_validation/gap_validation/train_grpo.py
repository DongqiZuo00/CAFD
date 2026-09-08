from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback, set_seed
from trl import GRPOConfig, GRPOTrainer

from .io import ARTIFACT_ROOT, load_yaml
from .modeling import enforce_text_policy_trainable, load_full_policy
from .prompting import pair_messages
from .splits import load_jsonl
from .verifiers import verify_math


class MilestoneSaveCallback(TrainerCallback):
    def __init__(self, milestones: set[int]):
        self.milestones = milestones

    def on_step_end(self, args, state, control, **kwargs):
        if int(state.global_step) in self.milestones:
            control.should_save = True
        return control


def math_reward(completions, solution, **kwargs):
    contents = []
    for completion in completions:
        if isinstance(completion, list):
            contents.append(completion[0]["content"])
        else:
            contents.append(str(completion))
    return [float(verify_math(text, gold).correct) for text, gold in zip(contents, solution)]


def dataset_for_math(path: Path) -> Dataset:
    rows: list[dict[str, Any]] = []
    for row in load_jsonl(path):
        rows.append({"prompt": pair_messages("math", row["prompt"]), "solution": row["solution"], "id": row["id"]})
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--max-completion-length", type=int, default=8192)
    parser.add_argument("--save-milestones", default="50")
    parser.add_argument("--deepspeed", type=Path, required=True)
    parser.add_argument("--prompts-per-device", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()

    set_seed(42)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    model = load_full_policy(args.model_id, args.revision, attn_implementation="sdpa")
    enforce_text_policy_trainable(model, args.output_dir / "trainable_parameters.json")
    milestones = {int(value) for value in args.save_milestones.split(",") if value}
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    prompts_per_update = args.prompts_per_device * world_size * args.gradient_accumulation_steps
    if prompts_per_update != 8:
        raise ValueError(
            "budget contract violation: prompts_per_device * world_size * gradient_accumulation_steps "
            f"must equal 8, observed {prompts_per_update}"
        )
    config = GRPOConfig(
        output_dir=str(args.output_dir),
        seed=42,
        data_seed=42,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        optim="adamw_torch_fused",
        per_device_train_batch_size=args.prompts_per_device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        generation_batch_size=32,
        num_generations=4,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        beta=0.0,
        loss_type="dapo",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        use_vllm=False,
        ds3_gather_for_generation=False,
        deepspeed=str(args.deepspeed),
        logging_steps=1,
        report_to="none",
        save_strategy="no",
        remove_unused_columns=False,
        shuffle_dataset=False,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[math_reward],
        args=config,
        train_dataset=dataset_for_math(args.dataset),
        processing_class=tokenizer,
        callbacks=[MilestoneSaveCallback(milestones)],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = args.output_dir / "final_model"
    trainer.save_model(str(final_dir))
    trainer.save_state()
    summary = {
        "model_id": args.model_id,
        "revision": args.revision,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "global_step": int(trainer.state.global_step),
        "rollout_slots_per_update": 32,
        "prompts_per_update": prompts_per_update,
        "responses_per_prompt": 4,
        "actual_world_size": world_size,
        "seed": 42,
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
