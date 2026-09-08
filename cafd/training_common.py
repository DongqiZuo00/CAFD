"""Shared deterministic generation, scoring, evaluation, and checkpoint helpers."""

from __future__ import annotations

import csv
import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    StopStringCriteria,
)

from .data import bounded_prompt, rollout_seed
from .exact_forward_kl import completion_prediction_mask
from .prompting import PROMPT_RENDERER_RAW, encode_prompt
from .verifier import score_completion, score_completion_details


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def primary(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistributedContext:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("formal CAFD training requires CUDA")
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    return DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size, device=torch.device("cuda", local_rank))


def barrier(context: DistributedContext) -> None:
    if context.world_size > 1:
        dist.barrier()


def seed_everything(seed: int, rank: int = 0) -> None:
    value = int(seed) + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


class HiddenCausalLM(nn.Module):
    """DDP-visible wrapper that returns hidden states without materializing full logits."""

    def __init__(self, causal_lm: PreTrainedModel) -> None:
        super().__init__()
        self.causal_lm = causal_lm

    @property
    def lm_head(self) -> nn.Module:
        return self.causal_lm.get_output_embeddings()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.causal_lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return output.last_hidden_state


def unwrap_hidden_model(model: HiddenCausalLM | DistributedDataParallel) -> HiddenCausalLM:
    return model.module if isinstance(model, DistributedDataParallel) else model


def trim_completion(tokens: torch.Tensor, eos_token_id: int | None, pad_token_id: int | None) -> list[int]:
    result: list[int] = []
    for token in tokens.tolist():
        if pad_token_id is not None and token == pad_token_id:
            break
        result.append(int(token))
        if eos_token_id is not None and token == eos_token_id:
            break
    return result


class ClosingFenceEOSProcessor(LogitsProcessor):
    """Force EOS immediately after a generated closing DSL fence."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, prompt_length: int, eos_token_id: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.eos_token_id = eos_token_id
        self.stop = StopStringCriteria(tokenizer, ["\n```"])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated = input_ids[:, self.prompt_length :]
        if generated.shape[1] == 0:
            return scores
        matched = self.stop(generated, scores)
        scores[matched] = float("-inf")
        scores[matched, self.eos_token_id] = 0.0
        return scores


@torch.no_grad()
def generate_group(
    model: HiddenCausalLM | DistributedDataParallel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    num_rollouts: int,
    max_prompt_length: int,
    max_completion_length: int,
    temperature: float,
    top_p: float,
    seed: int,
    sample: bool = True,
    prompt_renderer: str = PROMPT_RENDERER_RAW,
) -> tuple[list[int], list[list[int]]]:
    wrapper = unwrap_hidden_model(model)
    causal_lm = wrapper.causal_lm
    was_training = causal_lm.training
    causal_lm.eval()
    encoded = encode_prompt(
        tokenizer,
        prompt,
        renderer=prompt_renderer,
        max_length=max_prompt_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(next(causal_lm.parameters()).device)
    attention_mask = encoded["attention_mask"].to(input_ids.device)
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_completion_length,
        "num_return_sequences": num_rollouts,
        "do_sample": sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if tokenizer.eos_token_id is None:
        raise RuntimeError("fixed response contract requires an EOS token")
    kwargs["logits_processor"] = LogitsProcessorList(
        [ClosingFenceEOSProcessor(tokenizer, input_ids.shape[1], int(tokenizer.eos_token_id))]
    )
    if sample:
        kwargs.update(temperature=temperature, top_p=top_p)
    generated = causal_lm.generate(**kwargs)
    prompt_length = input_ids.shape[1]
    completions = [
        trim_completion(row[prompt_length:], tokenizer.eos_token_id, tokenizer.pad_token_id) for row in generated
    ]
    causal_lm.train(was_training)
    return input_ids[0].tolist(), completions


def make_sequence(
    prompt_ids: list[int], completion_ids: list[int], tokenizer: PreTrainedTokenizerBase, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(ids)
    prompt_lengths = torch.tensor([len(prompt_ids)], dtype=torch.long, device=device)
    mask = completion_prediction_mask(
        ids,
        attention,
        prompt_lengths,
        eos_token_id=tokenizer.eos_token_id,
    )
    return ids, attention, prompt_lengths, mask


def sampled_log_probs(
    hidden: torch.Tensor,
    head: nn.Module,
    input_ids: torch.Tensor,
    prediction_mask: torch.Tensor,
) -> torch.Tensor:
    selected_hidden = hidden[:, :-1][prediction_mask]
    labels = input_ids[:, 1:][prediction_mask]
    if selected_hidden.shape[0] == 0:
        return hidden.new_zeros((0,), dtype=torch.float32)
    logits = F.linear(selected_hidden, head.weight, getattr(head, "bias", None)).float()
    return F.log_softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)


def _all_reduce_int(value: int, context: DistributedContext) -> int:
    tensor = torch.tensor(value, dtype=torch.long, device=context.device)
    if context.world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def rlvr_update(
    model: HiddenCausalLM | DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    tokenizer: PreTrainedTokenizerBase,
    rows: list[dict[str, Any]],
    prompt_indices: list[int],
    *,
    update: int,
    rollouts_per_prompt: int,
    max_prompt_length: int,
    max_completion_length: int,
    temperature: float,
    top_p: float,
    score_mode: str,
    context: DistributedContext,
    prompt_renderer: str = PROMPT_RENDERER_RAW,
    clip_low: float = 0.2,
    clip_high: float = 0.3,
    raw_rollout_path: Path | None = None,
) -> dict[str, float | int]:
    wrapper = unwrap_hidden_model(model)
    rollouts: list[dict[str, Any]] = []
    generated_tokens = 0
    variable_reward_groups = 0
    for local_prompt_slot, row_index in enumerate(prompt_indices):
        row = rows[row_index]
        prompt_ids, completions = generate_group(
            model,
            tokenizer,
            bounded_prompt(row),
            num_rollouts=rollouts_per_prompt,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            temperature=temperature,
            top_p=top_p,
            seed=rollout_seed(update, context.rank * len(prompt_indices) + local_prompt_slot, 0),
            prompt_renderer=prompt_renderer,
        )
        rewards = [
            score_completion(tokenizer.decode(ids, skip_special_tokens=True), row["ground_truth"], score_mode)
            for ids in completions
        ]
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        mean = reward_tensor.mean()
        std = reward_tensor.std(unbiased=False)
        advantages = (reward_tensor - mean) / (std + 1e-4)
        variable_reward_groups += int(float(std.item()) > 0.0)
        for sample_slot, (completion_ids, reward, advantage) in enumerate(zip(completions, rewards, advantages, strict=True)):
            generated_tokens += len(completion_ids)
            rollouts.append(
                {
                    "prompt_ids": prompt_ids,
                    "completion_ids": completion_ids,
                    "reward": float(reward),
                    "advantage": float(advantage),
                    "row_id": row["id"],
                    "sample_slot": sample_slot,
                }
            )

    if raw_rollout_path is not None:
        import fcntl

        raw_rollout_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_rollout_path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            for item in rollouts:
                handle.write(
                    json.dumps(
                        {
                            "update": update + 1,
                            "rank": context.rank,
                            "row_id": item["row_id"],
                            "sample_slot": item["sample_slot"],
                            "completion_ids": item["completion_ids"],
                            "reward": item["reward"],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    local_positions = sum(len(item["completion_ids"]) for item in rollouts)
    global_positions = _all_reduce_int(local_positions, context)
    if global_positions == 0:
        raise RuntimeError("all RLVR rollouts were empty")
    optimizer.zero_grad(set_to_none=True)
    wrapper.causal_lm.train()
    loss_sum_value = 0.0
    for index, item in enumerate(rollouts):
        ids, attention, _, prediction_mask = make_sequence(
            item["prompt_ids"], item["completion_ids"], tokenizer, context.device
        )
        with torch.no_grad():
            old_hidden = wrapper(ids, attention)
            old_log_probs = sampled_log_probs(old_hidden, wrapper.lm_head, ids, prediction_mask).detach()
        sync_context = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and index != len(rollouts) - 1
            else nullcontext()
        )
        with sync_context:
            hidden = model(ids, attention)
            log_probs = sampled_log_probs(hidden, wrapper.lm_head, ids, prediction_mask)
            ratio = torch.exp(log_probs - old_log_probs)
            advantage = torch.tensor(item["advantage"], dtype=torch.float32, device=context.device)
            unclipped = ratio * advantage
            clipped = ratio.clamp(1.0 - clip_low, 1.0 + clip_high) * advantage
            loss_sum = -torch.minimum(unclipped, clipped).sum()
            scaled = loss_sum * context.world_size / global_positions
            scaled.backward()
            loss_sum_value += float(loss_sum.detach())
    torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
    optimizer.step()
    if hasattr(optimizer, "assert_fp32_states"):
        optimizer.assert_fp32_states()
    reward_sum = torch.tensor(sum(item["reward"] for item in rollouts), device=context.device)
    rollout_count = torch.tensor(len(rollouts), dtype=torch.long, device=context.device)
    generated = torch.tensor(generated_tokens, dtype=torch.long, device=context.device)
    variable_groups = torch.tensor(variable_reward_groups, dtype=torch.long, device=context.device)
    positive_rewards = torch.tensor(sum(item["reward"] > 0.0 for item in rollouts), dtype=torch.long, device=context.device)
    if context.world_size > 1:
        dist.all_reduce(reward_sum)
        dist.all_reduce(rollout_count)
        dist.all_reduce(generated)
        dist.all_reduce(variable_groups)
        dist.all_reduce(positive_rewards)
    return {
        "loss": loss_sum_value / max(local_positions, 1),
        "mean_reward": float(reward_sum.item() / max(rollout_count.item(), 1)),
        "generated_tokens": int(generated.item()),
        "completion_positions": global_positions,
        "variable_reward_groups": int(variable_groups.item()),
        "positive_reward_rollouts": int(positive_rewards.item()),
    }


@torch.no_grad()
def evaluate(
    model: HiddenCausalLM | DistributedDataParallel,
    tokenizer: PreTrainedTokenizerBase,
    rows: list[dict[str, Any]],
    *,
    max_prompt_length: int,
    max_completion_length: int,
    context: DistributedContext,
    prompt_renderer: str = PROMPT_RENDERER_RAW,
    include_details: bool = False,
) -> dict[str, int | float]:
    local_rows = rows[context.rank :: context.world_size]
    correct = 0
    generated_tokens = 0
    eos_terminated = 0
    token_limit_hits = 0
    tiers = {
        "invalid_format": 0,
        "format_only": 0,
        "parse_only": 0,
        "partial_pass": 0,
        "full_pass": 0,
    }
    for index, row in enumerate(local_rows):
        _, completions = generate_group(
            model,
            tokenizer,
            bounded_prompt(row),
            num_rollouts=1,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            temperature=1.0,
            top_p=1.0,
            seed=2027 + index + context.rank * 100_000,
            sample=False,
            prompt_renderer=prompt_renderer,
        )
        completion = completions[0]
        generated_tokens += len(completion)
        eos_terminated += int(
            bool(completion)
            and tokenizer.eos_token_id is not None
            and completion[-1] == tokenizer.eos_token_id
        )
        token_limit_hits += int(len(completion) == max_completion_length)
        details = score_completion_details(
            tokenizer.decode(completion, skip_special_tokens=True),
            row["ground_truth"],
            "hierarchical",
        )
        tier = str(details["tier"])
        tiers[tier] += 1
        correct += int(tier == "full_pass")
    tier_names = list(tiers)
    stats = torch.tensor(
        [correct, len(local_rows), generated_tokens, eos_terminated, token_limit_hits, *(tiers[name] for name in tier_names)],
        dtype=torch.long,
        device=context.device,
    )
    if context.world_size > 1:
        dist.all_reduce(stats)
    result: dict[str, int | float] = {
        "correct": int(stats[0].item()),
        "total": int(stats[1].item()),
        "accuracy": float(stats[0].item() / max(stats[1].item(), 1)),
        "generated_tokens": int(stats[2].item()),
    }
    if include_details:
        result["eos_terminated"] = int(stats[3].item())
        result["token_limit_hits"] = int(stats[4].item())
        for offset, name in enumerate(tier_names, start=5):
            result[name] = int(stats[offset].item())
        result["contract_valid"] = int(result["total"]) - int(result["invalid_format"])
        result["parse_or_better"] = sum(
            int(result[name]) for name in ("parse_only", "partial_pass", "full_pass")
        )
    return result


def checkpoint_complete(path: Path) -> bool:
    """Check that model shards and tokenizer payload were fully published."""
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    if not (path / "tokenizer_config.json").is_file():
        return False
    if not (path / "tokenizer.json").is_file():
        return False
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / index_name
        if not index_path.is_file():
            continue
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                return False
            shards = {path / str(name) for name in weight_map.values()}
        except (OSError, TypeError, ValueError):
            return False
        return all(
            shard.is_file() and shard.stat().st_size > 1_000_000
            for shard in shards
        )
    unsharded = [path / "model.safetensors", path / "pytorch_model.bin"]
    weights = [weight for weight in unsharded if weight.is_file()]
    return len(weights) == 1 and weights[0].stat().st_size > 1_000_000


def save_model_only(
    model: HiddenCausalLM | DistributedDataParallel,
    tokenizer: PreTrainedTokenizerBase,
    path: Path,
    context: DistributedContext,
) -> None:
    barrier(context)
    if context.primary and not checkpoint_complete(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / (
            f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
        )
        causal_lm = unwrap_hidden_model(model).causal_lm
        causal_lm.save_pretrained(
            temporary,
            safe_serialization=True,
            max_shard_size="4GB",
        )
        tokenizer.save_pretrained(temporary)
        if not checkpoint_complete(temporary):
            raise RuntimeError(f"incomplete checkpoint staging directory: {temporary}")
        if path.exists():
            aborted = path.parent / (
                f"{path.name}.aborted-{os.getpid()}-{time.time_ns()}"
            )
            os.replace(path, aborted)
        os.replace(temporary, path)
    barrier(context)


def append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", newline="", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0, os.SEEK_END)
        exists = handle.tell() > 0
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def snapshot_rng() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def wallclock() -> float:
    return time.time()
