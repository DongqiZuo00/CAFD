from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

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


def load_completion_dataset(path: Path) -> Dataset:
    rows: list[dict[str, Any]] = []
    for row in load_jsonl(path):
        rows.append({"prompt": pair_messages("math", row["prompt"]), "completion": row["completion"]})
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--revision", default="15852e8c16360a2fea060d615a32b45270f8a8fc")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--save-milestones", default="50")
    args = parser.parse_args()

    set_seed(42)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    model = load_full_policy(args.model_id, args.revision, attn_implementation="sdpa")
    enforce_text_policy_trainable(model, args.output_dir / "trainable_parameters.json")
    config = SFTConfig(
        output_dir=str(args.output_dir),
        seed=42,
        data_seed=42,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        optim="adamw_torch_fused",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
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
        callbacks=[MilestoneSaveCallback({int(value) for value in args.save_milestones.split(",") if value})],
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final_model"))
    trainer.save_state()
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "loss": "completion_only_cross_entropy",
                "global_sequence_batch_size": 32,
                "optimizer_updates": int(trainer.state.global_step),
                "seed": 42,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
