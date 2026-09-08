from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed

from .io import atomic_json
from .prompting import render
from .splits import load_jsonl
from .verifiers import verify_math


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def evaluate_item(item: dict[str, Any], dataset_path: Path, local_rank: int, seed: int) -> dict[str, Any]:
    predictions_path = Path(item["predictions"])
    summary_path = Path(item["summary"])
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    dataset = load_jsonl(dataset_path)
    existing = load_jsonl(predictions_path) if predictions_path.exists() else []
    completed = {str(row["prompt_id"]) for row in existing}

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(item["model"], revision=item.get("revision"))
    model = AutoModelForImageTextToText.from_pretrained(
        item["model"],
        revision=item.get("revision"),
        dtype=torch.bfloat16,
        device_map={"": local_rank},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = True
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = True
    model.eval()
    started_item = time.monotonic()
    for row in dataset:
        prompt_id = str(row.get("unique_id", row.get("id")))
        if prompt_id in completed:
            continue
        prompt = render(tokenizer, "qwen", "math", row["prompt"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        started = time.monotonic()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=int(item.get("max_new_tokens", 8192)),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        completion_ids = output[0, inputs.input_ids.shape[1] :]
        generation = tokenizer.decode(completion_ids, skip_special_tokens=False)
        result = verify_math(generation, row["answer"] if "answer" in row else row["solution"])
        append_jsonl(
            predictions_path,
            {
                "condition": item["condition"],
                "prompt_id": prompt_id,
                "raw_generation": generation,
                "parsed_answer": result.parsed_answer,
                "verifier_output": result.as_dict(),
                "reward": float(result.correct),
                "runtime_seconds": time.monotonic() - started,
                "generated_tokens": int(completion_ids.numel()),
                "correct": bool(result.correct),
            },
        )
        completed.add(prompt_id)
    predictions = load_jsonl(predictions_path)
    if len(predictions) != len(dataset):
        raise RuntimeError(
            f"incomplete evaluation for {item['condition']}: {len(predictions)} of {len(dataset)}"
        )
    correct = sum(bool(row["correct"]) for row in predictions)
    summary = {
        "condition": item["condition"],
        "model": item["model"],
        "revision": item.get("revision"),
        "checkpoint_step": int(item["checkpoint_step"]),
        "count": len(predictions),
        "correct": correct,
        "exact_answer_accuracy": correct / len(predictions),
        "generated_tokens": sum(int(row["generated_tokens"]) for row in predictions),
        "verifier_calls": len(predictions),
        "wall_time_seconds": sum(float(row["runtime_seconds"]) for row in predictions),
        "evaluation_process_seconds": time.monotonic() - started_item,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(local_rank)),
        "decoding": "deterministic_greedy",
        "seed": seed,
    }
    atomic_json(summary_path, summary)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation requires CUDA")
    torch.cuda.set_device(local_rank)
    for index, item in enumerate(plan["items"]):
        if index % world_size == rank:
            evaluate_item(item, Path(plan["dataset"]), local_rank, int(plan.get("seed", 42)))


if __name__ == "__main__":
    main()
