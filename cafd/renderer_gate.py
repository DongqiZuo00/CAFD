"""Low-cost causal A/B gate for the pinned Teacher prompt renderer."""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml

from .data import bounded_prompt, load_rows, prompt_stream, rollout_seed
from .modeling import TEACHER_ID, TEACHER_REVISION, load_model, load_tokenizers
from .prompting import (
    PROMPT_RENDERER_QWEN3_INSTRUCT,
    PROMPT_RENDERER_RAW,
    assert_matches_teacher_chat_template,
)
from .training_common import HiddenCausalLM, generate_group, init_distributed, seed_everything, write_json
from .verifier import score_completion_details


def _summarize(
    model: HiddenCausalLM,
    tokenizer,
    rows: list[dict[str, Any]],
    indices: list[int],
    *,
    renderer: str,
    rollouts_per_prompt: int,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> dict[str, Any]:
    tiers: Counter[str] = Counter()
    lengths: list[int] = []
    rewards: list[float] = []
    variable_groups = 0
    for prompt_slot, row_index in enumerate(indices):
        row = rows[row_index]
        _, completions = generate_group(
            model,
            tokenizer,
            bounded_prompt(row),
            num_rollouts=rollouts_per_prompt,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=rollout_seed(0, prompt_slot, 0, base_seed=seed),
            sample=True,
            prompt_renderer=renderer,
        )
        group_rewards: list[float] = []
        for completion_ids in completions:
            details = score_completion_details(
                tokenizer.decode(completion_ids, skip_special_tokens=True),
                row["ground_truth"],
                "hierarchical",
                require_contract=True,
            )
            reward = float(details["reward"])
            tiers[str(details["tier"])] += 1
            lengths.append(len(completion_ids))
            rewards.append(reward)
            group_rewards.append(reward)
        variable_groups += int(max(group_rewards) > min(group_rewards))
    format_or_better = sum(tiers[key] for key in ("format_only", "parse_only", "partial_pass", "full_pass"))
    parse_or_better = sum(tiers[key] for key in ("parse_only", "partial_pass", "full_pass"))
    return {
        "renderer": renderer,
        "rollouts": len(rewards),
        "tiers": dict(sorted(tiers.items())),
        "format_or_better": format_or_better,
        "parse_or_better": parse_or_better,
        "partial_or_better": tiers["partial_pass"] + tiers["full_pass"],
        "full_pass": tiers["full_pass"],
        "positive_reward_rollouts": sum(reward > 0.0 for reward in rewards),
        "variable_reward_groups": variable_groups,
        "mean_reward": statistics.fmean(rewards),
        "mean_tokens": statistics.fmean(lengths),
        "median_tokens": statistics.median(lengths),
        "max_tokens": max(lengths),
        "token_limit_hits": sum(length >= max_new_tokens for length in lengths),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = yaml.safe_load((root / "configs" / "cafd" / "manufactoria_has.yaml").read_text(encoding="utf-8"))
    gate = config["renderer_gate"]
    generation = config["generation"]
    context = init_distributed()
    if context.world_size != 1:
        raise RuntimeError("renderer hypothesis gate requires exactly one GPU")
    seed_everything(int(config["seed"]))
    cache = root / ".cache" / "huggingface" / "hub"
    teacher_tokenizer, _ = load_tokenizers(cache)
    assert_matches_teacher_chat_template(teacher_tokenizer)
    causal_lm = load_model(
        TEACHER_ID,
        TEACHER_REVISION,
        cache_dir=cache,
        device=context.device,
        trainable=False,
    )
    model = HiddenCausalLM(causal_lm)
    rows = load_rows(root, "train")
    indices = prompt_stream(rows, 1, int(gate["prompts"]), int(config["seed"]))[0]
    common = {
        "model": model,
        "tokenizer": teacher_tokenizer,
        "rows": rows,
        "indices": indices,
        "rollouts_per_prompt": int(gate["rollouts_per_prompt"]),
        "max_prompt_length": int(config["teacher_route"]["max_prompt_length"]),
        "max_new_tokens": int(generation["max_new_tokens"]),
        "temperature": float(config["teacher_route"]["temperature"]),
        "top_p": float(config["teacher_route"]["top_p"]),
        "seed": int(config["seed"]),
    }
    raw = _summarize(renderer=PROMPT_RENDERER_RAW, **common)
    chat = _summarize(renderer=PROMPT_RENDERER_QWEN3_INSTRUCT, **common)
    passed = (
        chat["format_or_better"] > raw["format_or_better"]
        and chat["parse_or_better"] > raw["parse_or_better"]
        and chat["positive_reward_rollouts"] > raw["positive_reward_rollouts"]
        and chat["variable_reward_groups"] >= int(gate["minimum_variable_reward_groups"])
        and chat["mean_tokens"] > raw["mean_tokens"]
    )
    payload = {
        "status": "passed" if passed else "RENDERER_HYPOTHESIS_FAILED",
        "hypothesis": "official_instruct_chat_framing_improves_T0_contract_and_reward_support",
        "same_model_prompts_seeds_sampling": True,
        "raw": raw,
        "chat": chat,
    }
    output = root / "artifacts" / "cafd" / "renderer_gate_chat_hier_v3.json"
    write_json(output, payload)
    write_json(root / "state" / "cafd" / "renderer_gate_chat_hier_v3.json", payload)
    del model, causal_lm
    torch.cuda.empty_cache()
    if not passed:
        raise SystemExit(45)


if __name__ == "__main__":
    main()
