from __future__ import annotations

import json

from trl import DistillationConfig, GRPOConfig, SFTConfig

from .io import ARTIFACT_ROOT, atomic_json


def main() -> None:
    grpo = GRPOConfig(
        output_dir="/tmp/gap-config-contract-grpo",
        seed=42,
        data_seed=42,
        max_steps=50,
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        optim="adamw_torch_fused",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        generation_batch_size=32,
        num_generations=4,
        max_completion_length=8192,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        beta=0.0,
        loss_type="dapo",
        use_cpu=True,
        bf16=False,
        tf32=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        use_vllm=False,
        ds3_gather_for_generation=False,
        deepspeed="configs/gap_validation/deepspeed_zero3.json",
        logging_steps=1,
        report_to="none",
        save_strategy="no",
        remove_unused_columns=False,
        shuffle_dataset=False,
    )
    distillation = DistillationConfig(
        output_dir="/tmp/gap-config-contract-gkd",
        max_steps=50,
        learning_rate=1e-6,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        max_completion_length=8192,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        beta=0.0,
        use_cpu=True,
        bf16=False,
        tf32=False,
        save_strategy="no",
        shuffle_dataset=False,
    )
    sft = SFTConfig(
        output_dir="/tmp/gap-config-contract-sft",
        max_steps=50,
        learning_rate=1e-6,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        max_length=16384,
        completion_only_loss=True,
        use_cpu=True,
        bf16=False,
        tf32=False,
        save_strategy="no",
        shuffle_dataset=False,
    )
    result = {
        "status": "VALID",
        "grpo": {"num_generations": grpo.num_generations, "generation_batch_size": grpo.generation_batch_size},
        "distillation": {"beta": distillation.beta, "temperature": distillation.temperature},
        "sft": {"completion_only_loss": sft.completion_only_loss, "max_length": sft.max_length},
    }
    atomic_json(ARTIFACT_ROOT / "config_contract.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
