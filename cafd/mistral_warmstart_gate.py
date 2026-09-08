"""Verify development ability and nonzero hierarchical reward support before RLVR."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .configuration import load_experiment_config
from .data import load_rows, prompt_stream
from .layout import experiment_layout
from .mistral_disjoint import teacher_rlvr_rows
from .mistral_runtime import (
    assert_exact_tokenizer_pair,
    load_model,
    load_tokenizers,
    validate_config,
)
from .renderer_gate import _summarize
from .training_common import HiddenCausalLM, evaluate, init_distributed, seed_everything, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = load_experiment_config(root)
    validate_config(config)
    context = init_distributed()
    if context.world_size != 1:
        raise RuntimeError("Mistral warm-start gate requires exactly one GPU")
    seed_everything(int(config["seed"]))
    layout = experiment_layout(root, config)
    route = config["teacher_route"]
    generation = config["generation"]
    checkpoint = (root / str(route["initial_checkpoint"])).resolve()
    cache = root / ".cache" / "huggingface" / "hub"
    teacher_tokenizer, student_tokenizer = load_tokenizers(cache)
    causal_lm = load_model(
        str(checkpoint),
        "",
        cache_dir=cache,
        device=context.device,
        trainable=False,
    )
    assert_exact_tokenizer_pair(
        teacher_tokenizer,
        student_tokenizer,
        teacher_model=causal_lm,
    )
    model = HiddenCausalLM(causal_lm)
    development = evaluate(
        model,
        teacher_tokenizer,
        load_rows(root, "development"),
        max_prompt_length=int(route["max_prompt_length"]),
        max_completion_length=int(generation["max_new_tokens"]),
        context=context,
        prompt_renderer=str(generation["prompt_renderer"]),
        include_details=True,
    )
    gate = config["warmstart_gate"]
    rows = teacher_rlvr_rows(root, config)
    indices = prompt_stream(
        rows, 1, int(gate["prompts"]), int(config["seed"])
    )[0]
    support = _summarize(
        model,
        teacher_tokenizer,
        rows,
        indices,
        renderer=str(generation["prompt_renderer"]),
        rollouts_per_prompt=int(gate["rollouts_per_prompt"]),
        max_prompt_length=int(route["max_prompt_length"]),
        max_new_tokens=int(generation["max_new_tokens"]),
        temperature=float(route["temperature"]),
        top_p=float(route["top_p"]),
        seed=int(config["seed"]),
    )
    limit_rate = support["token_limit_hits"] / support["rollouts"]
    passed = (
        int(development["correct"]) >= int(gate["minimum_development_correct"])
        and int(support["format_or_better"]) >= int(gate["minimum_format_or_better"])
        and int(support["parse_or_better"]) >= int(gate["minimum_parse_or_better"])
        and int(support["partial_or_better"]) >= int(gate["minimum_partial_or_better"])
        and int(support["positive_reward_rollouts"]) >= int(gate["minimum_positive_reward_rollouts"])
        and int(support["variable_reward_groups"]) >= int(gate["minimum_variable_reward_groups"])
        and limit_rate <= float(gate["maximum_token_limit_rate"])
    )
    payload = {
        "status": "passed" if passed else "MISTRAL_WARMSTART_GATE_FAILED",
        "checkpoint": str(checkpoint),
        "development": development,
        "hierarchical_reward_support": support,
        "token_limit_rate": limit_rate,
        "frozen_test_accessed": False,
    }
    write_json(layout.artifact_root / "mistral_warmstart_gate.json", payload)
    write_json(layout.state_root / "mistral_warmstart_gate.json", payload)
    del model, causal_lm
    torch.cuda.empty_cache()
    if not passed:
        raise SystemExit(46)


if __name__ == "__main__":
    main()
