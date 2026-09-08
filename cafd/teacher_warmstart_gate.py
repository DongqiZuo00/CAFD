"""Gate the SFT25 warm-start on development and real rollout reward support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .configuration import load_experiment_config
from .data import load_rows, prompt_stream
from .modeling import assert_exact_tokenizer_pair, load_model, load_tokenizers
from .renderer_gate import _summarize
from .training_common import HiddenCausalLM, evaluate, init_distributed, seed_everything, write_json


def clopper_pearson_upper(hits: int, total: int, alpha: float = 0.05) -> float:
    if not 0 <= hits <= total or total <= 0:
        raise ValueError((hits, total))
    if hits == total:
        return 1.0
    low = hits / total
    high = 1.0
    for _ in range(100):
        probability = (low + high) / 2.0
        cdf = sum(
            __import__("math").comb(total, index)
            * probability**index
            * (1.0 - probability) ** (total - index)
            for index in range(hits + 1)
        )
        if cdf > alpha:
            low = probability
        else:
            high = probability
    return high


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = load_experiment_config(root)
    if config["protocol"] != "compute_bounded_v4_sft25_warmstart":
        raise RuntimeError(f"unexpected protocol: {config['protocol']}")
    context = init_distributed()
    if context.world_size != 1:
        raise RuntimeError("warm-start gate requires exactly one GPU")
    seed_everything(int(config["seed"]))
    route = config["teacher_route"]
    generation = config["generation"]
    checkpoint = (root / str(route["initial_checkpoint"])).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    cache = root / ".cache" / "huggingface" / "hub"
    teacher_tokenizer, student_tokenizer = load_tokenizers(cache)
    causal_lm = load_model(str(checkpoint), "", cache_dir=cache, device=context.device, trainable=False)
    assert_exact_tokenizer_pair(teacher_tokenizer, student_tokenizer, teacher_model=causal_lm)
    model = HiddenCausalLM(causal_lm)
    development = evaluate(
        model, teacher_tokenizer, load_rows(root, "development"),
        max_prompt_length=int(route["max_prompt_length"]),
        max_completion_length=int(generation["max_new_tokens"]),
        context=context,
        prompt_renderer=str(generation["prompt_renderer"]),
    )
    rows = load_rows(root, "train")
    gate = config["warmstart_gate"]
    indices = prompt_stream(rows, 1, int(gate["prompts"]), int(config["seed"]))[0]
    support = _summarize(
        model, teacher_tokenizer, rows, indices,
        renderer=str(generation["prompt_renderer"]),
        rollouts_per_prompt=int(gate["rollouts_per_prompt"]),
        max_prompt_length=int(route["max_prompt_length"]),
        max_new_tokens=int(generation["max_new_tokens"]),
        temperature=float(route["temperature"]),
        top_p=float(route["top_p"]),
        seed=int(config["seed"]),
    )
    tiers = support["tiers"]
    canonical_path = (root / str(gate["canonical_replay"])).resolve()
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    token_limit_upper = clopper_pearson_upper(
        int(support["token_limit_hits"]),
        int(support["rollouts"]),
    )
    passed = (
        int(canonical["passed"]) == int(canonical["total"]) == 742
        and int(canonical["truncated"]) == 0
        and int(canonical["max_tokens"]) < int(canonical["max_new_tokens"])
        and int(canonical["loss_mask_exact"]) == 678
        and int(canonical["stored_completion_exact"]) == 678
        and int(development["correct"]) == int(gate["expected_development_correct"])
        and int(development["total"]) == 64
        and int(support["full_pass"]) >= int(gate["minimum_full_pass"])
        and int(tiers.get("partial_pass", 0)) >= int(gate["minimum_partial_pass"])
        and int(support["variable_reward_groups"]) >= int(gate["minimum_variable_reward_groups"])
        and int(support["positive_reward_rollouts"]) > 0
        and token_limit_upper < float(gate["maximum_token_limit_cp95_upper"])
    )
    payload = {
        "status": "passed" if passed else "WARMSTART_GATE_FAILED",
        "checkpoint": str(checkpoint),
        "canonical_replay": {"path": str(canonical_path), **canonical},
        "development": development,
        "reward_support": support,
        "sampled_length_audit": {
            "hits": int(support["token_limit_hits"]),
            "total": int(support["rollouts"]),
            "one_sided_95pct_clopper_pearson_upper": token_limit_upper,
            "required_upper_exclusive": float(gate["maximum_token_limit_cp95_upper"]),
        },
    }
    run_id = config["experiment"]["run_id"]
    write_json(root / "artifacts" / "cafd" / "experiments" / run_id / "warmstart_gate.json", payload)
    write_json(root / "state" / "cafd" / "experiments" / run_id / "warmstart_gate.json", payload)
    del model, causal_lm
    torch.cuda.empty_cache()
    if not passed:
        raise SystemExit(46)


if __name__ == "__main__":
    main()
