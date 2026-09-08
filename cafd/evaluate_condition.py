"""One-time official-test evaluation of an already frozen condition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from .configuration import load_experiment_config
from .data import load_rows
from .layout import experiment_layout
from .modeling import assert_exact_tokenizer_pair, load_model, load_tokenizers
from .training_common import HiddenCausalLM, evaluate, init_distributed, write_json


def _final_record(
    condition: str,
    frozen_record: dict,
    stats: dict,
) -> dict:
    """Keep frozen training costs separate from held-out inference costs."""
    return {
        **stats,
        **frozen_record,
        "condition": condition,
        "evaluation_generated_tokens": int(stats.get("generated_tokens", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = load_experiment_config(root)
    layout = experiment_layout(root, config)
    frozen = json.loads(layout.frozen_selection.read_text(encoding="utf-8"))
    if args.condition not in frozen["conditions"]:
        raise KeyError(args.condition)
    output = layout.final / f"{args.condition}.json"
    if output.exists():
        return
    context = init_distributed()
    if context.world_size != 1:
        raise RuntimeError("final condition evaluation expects one GPU")
    record = frozen["conditions"][args.condition]
    cache = root / ".cache" / "huggingface" / "hub"
    teacher_tokenizer, student_tokenizer = load_tokenizers(cache)
    tokenizer = AutoTokenizer.from_pretrained(record["checkpoint"], padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Either checkpoint type must still be the exact shared pair.
    assert_exact_tokenizer_pair(teacher_tokenizer, tokenizer)
    assert_exact_tokenizer_pair(student_tokenizer, tokenizer)
    model = load_model(record["checkpoint"], "", cache_dir=cache, device=context.device, trainable=False)
    assert_exact_tokenizer_pair(teacher_tokenizer, student_tokenizer, teacher_model=model)
    max_completion = int(config["generation"]["max_new_tokens"])
    stats = evaluate(
        HiddenCausalLM(model),
        tokenizer,
        load_rows(root, "test"),
        max_prompt_length=2048,
        max_completion_length=max_completion,
        context=context,
        prompt_renderer=str(config["generation"]["prompt_renderer"]),
    )
    write_json(output, _final_record(args.condition, record, stats))


if __name__ == "__main__":
    main()
