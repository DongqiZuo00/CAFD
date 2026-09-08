"""Lowest-cost frozen diagnostic for the blocked CAFD capacity control."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml

from .canonical import canonical_program, fenced
from .data import bounded_prompt, load_rows
from .exact_forward_kl import completion_prediction_mask
from .modeling import assert_exact_tokenizer_pair, load_model, load_tokenizers
from .training_common import HiddenCausalLM, generate_group, init_distributed, seed_everything, write_json
from .verifier import extract_contract_program, score_completion, verify_program


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _gold_replay(root: Path, tokenizer, max_new_tokens: int) -> dict[str, Any]:
    stored = {
        row["id"]: row
        for row in _read_jsonl(root / "runs" / "cafd" / "capacity" / "verified_train_solutions.jsonl")
    }
    totals = Counter()
    failures: list[dict[str, Any]] = []
    maximum_tokens = 0
    for split in ("train", "development"):
        rows = load_rows(root, split)
        for row in rows:
            totals["total"] += 1
            completion = fenced(canonical_program(row))
            completion_ids = tokenizer.encode(completion, add_special_tokens=False)
            maximum_tokens = max(maximum_tokens, len(completion_ids))
            truncated = len(completion_ids) > max_new_tokens
            roundtrip = tokenizer.decode(completion_ids + [tokenizer.eos_token_id], skip_special_tokens=True)
            passed = score_completion(roundtrip, row["ground_truth"], "full_pass", require_contract=True) == 1.0
            totals["truncated"] += int(truncated)
            totals["passed"] += int(passed)
            if split == "train":
                record = stored.get(row["id"])
                stored_ok = record is not None
                prompt_ids = tokenizer.encode(bounded_prompt(row), add_special_tokens=True)
                stored_prompt_ok = stored_ok and record["prompt_ids"] == prompt_ids
                stored_completion = tokenizer.decode(record["completion_ids"], skip_special_tokens=True) if stored_ok else ""
                stored_completion_ok = stored_ok and stored_completion == roundtrip
                if stored_ok:
                    ids = torch.tensor([record["prompt_ids"] + record["completion_ids"]], dtype=torch.long)
                    attention = torch.ones_like(ids)
                    prompt_lengths = torch.tensor([len(record["prompt_ids"])], dtype=torch.long)
                    mask = completion_prediction_mask(ids, attention, prompt_lengths, eos_token_id=tokenizer.eos_token_id)
                    mask_ok = int(mask.sum().item()) == len(record["completion_ids"])
                else:
                    mask_ok = False
                totals["stored_prompt_ok"] += int(stored_prompt_ok)
                totals["stored_completion_ok"] += int(stored_completion_ok)
                totals["loss_mask_ok"] += int(mask_ok)
            if truncated or not passed:
                failures.append({"split": split, "id": row["id"], "tokens": len(completion_ids), "truncated": truncated, "passed": passed})
    return {
        "total": totals["total"],
        "passed": totals["passed"],
        "truncated": totals["truncated"],
        "max_tokens": maximum_tokens,
        "max_new_tokens": max_new_tokens,
        "stored_training_records": len(stored),
        "stored_prompt_exact": totals["stored_prompt_ok"],
        "stored_completion_exact": totals["stored_completion_ok"],
        "loss_mask_exact": totals["loss_mask_ok"],
        "failures": failures,
    }


def _generate_split(model, tokenizer, rows, split: str, max_new_tokens: int, output: Path) -> None:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if split == "development":
            old_rank = index % 4
            old_local_index = index // 4
            seed = 2027 + old_local_index + old_rank * 100_000
        else:
            seed = 2027 + index
        _, completions = generate_group(
            model,
            tokenizer,
            bounded_prompt(row),
            num_rollouts=1,
            max_prompt_length=2048,
            max_completion_length=max_new_tokens,
            temperature=1.0,
            top_p=1.0,
            seed=seed,
            sample=False,
        )
        completion_ids = completions[0]
        records.append(
            {
                "split": split,
                "index": index,
                "id": row["id"],
                "completion": tokenizer.decode(completion_ids, skip_special_tokens=True),
                "completion_ids": completion_ids,
                "token_count": len(completion_ids),
                "ended_with_eos": bool(completion_ids and completion_ids[-1] == tokenizer.eos_token_id),
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(rows):
            _write_jsonl(output, records)


def _offline_replay(path: Path, row_map: dict[str, dict[str, Any]], max_new_tokens: int) -> dict[str, Any]:
    categories = Counter()
    details: list[dict[str, Any]] = []
    tokens = 0
    for record in _read_jsonl(path):
        completion = record["completion"]
        tokens += int(record["token_count"])
        hit_limit = int(record["token_count"]) >= max_new_tokens and not record["ended_with_eos"]
        program = extract_contract_program(completion)
        if hit_limit:
            category = "truncation"
        elif program is None:
            category = "format"
        else:
            verification = verify_program(program, row_map[record["id"]]["ground_truth"])
            if not verification["valid"]:
                category = "parse"
            elif verification["all_passed"]:
                category = "success"
            else:
                category = "semantic"
        categories[category] += 1
        details.append({"id": record["id"], "category": category, "token_count": record["token_count"]})
    total = sum(categories.values())
    return {
        "total": total,
        "correct": categories["success"],
        "accuracy": categories["success"] / total if total else 0.0,
        "failure_categories": {key: categories[key] for key in ("format", "parse", "truncation", "semantic")},
        "generated_tokens": tokens,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = yaml.safe_load((root / "configs" / "cafd" / "manufactoria_has.yaml").read_text(encoding="utf-8"))
    max_new_tokens = int(config["generation"]["max_new_tokens"])
    output = root / "artifacts" / "cafd" / "capacity_diagnosis"
    output.mkdir(parents=True, exist_ok=True)
    teacher_tokenizer, student_tokenizer = load_tokenizers(root / ".cache" / "huggingface" / "hub")
    assert_exact_tokenizer_pair(teacher_tokenizer, student_tokenizer)

    gold = _gold_replay(root, student_tokenizer, max_new_tokens)
    write_json(output / "gold_replay.json", gold)
    gold_ok = (
        gold["passed"] == gold["total"] == 742
        and gold["truncated"] == 0
        and gold["stored_training_records"] == 678
        and gold["stored_prompt_exact"] == 678
        and gold["stored_completion_exact"] == 678
        and gold["loss_mask_exact"] == 678
    )
    if not gold_ok:
        write_json(output / "diagnosis.json", {"status": "GOLD_OR_PIPELINE_INVALID", "gold": gold})
        raise SystemExit(44)

    context = init_distributed()
    if context.world_size != 1:
        raise RuntimeError("capacity diagnostic must use exactly one GPU")
    seed_everything(config["seed"])
    checkpoint = root / "runs" / "cafd" / "capacity" / "SFT350"
    causal_lm = load_model(str(checkpoint), "", cache_dir=root / ".cache" / "huggingface" / "hub", device=context.device, trainable=False)
    assert_exact_tokenizer_pair(teacher_tokenizer, student_tokenizer, student_model=causal_lm)
    model = HiddenCausalLM(causal_lm)
    development_rows = load_rows(root, "development")
    training_rows = load_rows(root, "train")
    development_path = output / "sft350_development_outputs.jsonl"
    training_path = output / "sft350_training_outputs.jsonl"
    _generate_split(model, student_tokenizer, development_rows, "development", max_new_tokens, development_path)
    _generate_split(model, student_tokenizer, training_rows, "train", max_new_tokens, training_path)
    del model, causal_lm
    torch.cuda.empty_cache()

    development = _offline_replay(development_path, {row["id"]: row for row in development_rows}, max_new_tokens)
    training = _offline_replay(training_path, {row["id"]: row for row in training_rows}, max_new_tokens)
    reproduced = development["correct"] == 3 and development["total"] == 64
    if not reproduced:
        status = "DEVELOPMENT_REPRODUCTION_MISMATCH"
    elif training["accuracy"] >= 52 / 64:
        status = "HELD_OUT_GENERALIZATION_BLOCKED"
    else:
        status = "SFT_PIPELINE_OR_OPTIMIZATION_BLOCKED"
    write_json(
        output / "diagnosis.json",
        {
            "status": status,
            "checkpoint": str(checkpoint.resolve()),
            "gold": gold,
            "development_reproduced_3_of_64": reproduced,
            "development": development,
            "training": training,
        },
    )


if __name__ == "__main__":
    main()
