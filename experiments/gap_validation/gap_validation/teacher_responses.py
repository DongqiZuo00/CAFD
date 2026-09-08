from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed

from .io import atomic_json
from .prompting import render
from .splits import load_jsonl
from .verifiers import verify_math


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def stable_prompt_seed(problem_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{problem_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def prompt_shard(stream_position: int, num_shards: int) -> int:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    return stream_position % num_shards


def worker_output_dir(root: Path, shard_index: int, num_shards: int) -> Path:
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    if num_shards == 1:
        return root
    return root / "shards" / f"{shard_index:04d}-of-{num_shards:04d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/frozen/math_training/student_prompt_stream.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--responses-per-prompt", type=int, default=8)
    parser.add_argument("--required-successes", type=int, default=6400)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard-index", type=int)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    num_shards = args.num_shards or world_size
    shard_index = args.shard_index if args.shard_index is not None else rank
    if world_size > 1 and num_shards != world_size:
        raise ValueError("torchrun world size must equal --num-shards")
    output_dir = worker_output_dir(args.output_dir, shard_index, num_shards)

    set_seed(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "all_candidates.jsonl"
    successes_path = output_dir / "teacher_successes.jsonl"
    existing = load_jsonl(candidates_path) if candidates_path.exists() else []
    completed_ids = {str(row["problem_id"]) for row in existing}
    success_count = sum(bool(row.get("correct")) for row in existing)
    generated_tokens = sum(int(row.get("generated_tokens", 0)) for row in existing)
    verifier_calls = len(existing)

    if not torch.cuda.is_available():
        raise RuntimeError("Teacher response generation requires CUDA")
    torch.cuda.set_device(local_rank)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model),
        dtype=torch.bfloat16,
        device_map={"": local_rank},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    for row in load_jsonl(args.dataset):
        problem_id = str(row["id"])
        stream_position = int(row["stream_position"])
        if prompt_shard(stream_position, num_shards) != shard_index:
            continue
        target_reached = args.required_successes > 0 and success_count >= args.required_successes
        if problem_id in completed_ids or target_reached:
            continue
        prompt_seed = stable_prompt_seed(problem_id, args.seed)
        set_seed(prompt_seed)
        serialized = render(tokenizer, "qwen", "math", row["prompt"])
        inputs = tokenizer(serialized, return_tensors="pt").to(model.device)
        started = time.monotonic()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                top_k=20,
                num_return_sequences=args.responses_per_prompt,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        for slot, output in enumerate(outputs):
            completion_ids = output[inputs.input_ids.shape[1] :]
            generation = tokenizer.decode(completion_ids, skip_special_tokens=False)
            result = verify_math(generation, row["solution"])
            record = {
                "problem_id": problem_id,
                "stream_position": stream_position,
                "rollout_slot": slot,
                "prompt_seed": prompt_seed,
                "prompt": row["prompt"],
                "completion": generation,
                "correct": result.correct,
                "verifier_output": result.as_dict(),
                "generated_tokens": int(completion_ids.numel()),
                "runtime_seconds_per_prompt_batch": time.monotonic() - started,
            }
            append_jsonl(candidates_path, record)
            verifier_calls += 1
            generated_tokens += record["generated_tokens"]
            if result.correct:
                append_jsonl(successes_path, record)
                success_count += 1
        completed_ids.add(problem_id)
        atomic_json(
            output_dir / "status.json",
            {
                "state": "RUNNING",
                "shard_index": shard_index,
                "num_shards": num_shards,
                "completed_prompts": len(completed_ids),
                "successful_responses": success_count,
                "required_successes": args.required_successes,
                "generated_tokens": generated_tokens,
                "verifier_calls": verifier_calls,
            },
        )
    state = (
        "COMPLETED"
        if args.required_successes == 0 or success_count >= args.required_successes
        else "RFT_DATA_INSUFFICIENT"
    )
    status = {
        "state": state,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "completed_prompts": len(completed_ids),
        "successful_responses": success_count,
        "required_successes": args.required_successes,
        "generated_tokens": generated_tokens,
        "verifier_calls": verifier_calls,
    }
    atomic_json(output_dir / "status.json", status)
    print(json.dumps(status, indent=2))
    if state != "COMPLETED":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
