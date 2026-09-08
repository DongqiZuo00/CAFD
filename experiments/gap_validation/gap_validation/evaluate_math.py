from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from .prompting import render
from .splits import load_jsonl
from .verifiers import verify_math


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    predictions = []
    for row in load_jsonl(args.dataset):
        prompt = render(tokenizer, "qwen", "math", row["prompt"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        started = time.monotonic()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generation = tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=False)
        result = verify_math(generation, row["answer"] if "answer" in row else row["solution"])
        predictions.append(
            {
                "prompt_id": row.get("unique_id", row.get("id")),
                "raw_generation": generation,
                "parsed_answer": result.parsed_answer,
                "verifier_output": result.as_dict(),
                "reward": float(result.correct),
                "runtime_seconds": time.monotonic() - started,
                "correct": result.correct,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n", encoding="utf-8")
    score = sum(row["correct"] for row in predictions) / len(predictions) if predictions else 0.0
    summary = {"count": len(predictions), "exact_answer_accuracy": score}
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
