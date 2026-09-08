from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .io import ARTIFACT_ROOT, atomic_json, load_yaml
from .prompting import render


AUDIT_STRINGS = [
    "ASCII normalization test: A  B\\nC",
    "Unicode: café, naïve, 中文, π, ≤, \u00a0",
    "Code: def f(x: int) -> int:\\n    return x ** 2",
    "Math: $\\frac{1}{2}+\\sqrt{2}$ and \\boxed{3}",
]


def _vocab_digest(vocab: dict[str, int]) -> str:
    payload = json.dumps(sorted(vocab.items()), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _normalizer(tokenizer: Any) -> str | None:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    normalizer = getattr(backend, "normalizer", None)
    return None if normalizer is None else str(normalizer)


def describe(tokenizer: Any) -> dict[str, Any]:
    vocab = tokenizer.get_vocab()
    return {
        "class": type(tokenizer).__name__,
        "vocabulary_size": len(vocab),
        "vocabulary_sha256": _vocab_digest(vocab),
        "added_tokens": sorted(tokenizer.get_added_vocab().items()),
        "special_ids": {
            key: getattr(tokenizer, f"{key}_token_id", None)
            for key in ("bos", "eos", "pad", "unk")
        },
        "normalizer": _normalizer(tokenizer),
        "actual_tokenization": {
            text: tokenizer(text, add_special_tokens=False).input_ids for text in AUDIT_STRINGS
        },
    }


def audit_pair(name: str, pair: dict[str, Any]) -> dict[str, Any]:
    if pair.get("access") != "available":
        return {"pair": name, "status": "BLOCKED", "reason": pair.get("blocker")}
    loaded: dict[str, Any] = {}
    descriptions: dict[str, Any] = {}
    for role in ("teacher", "student"):
        spec = pair[role]
        tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
        loaded[role] = tokenizer
        descriptions[role] = describe(tokenizer)
    teacher_vocab = loaded["teacher"].get_vocab()
    student_vocab = loaded["student"].get_vocab()
    mapping_equal = teacher_vocab == student_vocab
    prompt_checks: list[dict[str, Any]] = []
    for task in ("code", "math"):
        serialized = render(loaded["teacher"], name, task, AUDIT_STRINGS[-1])
        teacher_ids = loaded["teacher"](serialized, add_special_tokens=False).input_ids
        student_ids = loaded["student"](serialized, add_special_tokens=False).input_ids
        prompt_checks.append(
            {
                "task": task,
                "serialized_prompt_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
                "teacher_ids_sha256": hashlib.sha256(json.dumps(teacher_ids).encode()).hexdigest(),
                "student_ids_sha256": hashlib.sha256(json.dumps(student_ids).encode()).hexdigest(),
                "ids_equal": teacher_ids == student_ids,
                "length": len(teacher_ids),
            }
        )
    scalar_equal = all(
        descriptions["teacher"][key] == descriptions["student"][key]
        for key in ("vocabulary_size", "added_tokens", "special_ids", "normalizer", "actual_tokenization")
    )
    valid = mapping_equal and scalar_equal and all(item["ids_equal"] for item in prompt_checks)
    return {
        "pair": name,
        "status": "VALID" if valid else "INVALID_TOKENIZER_PAIR",
        "complete_mapping_equal": mapping_equal,
        "descriptions": descriptions,
        "prompt_checks": prompt_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "tokenizer_audit.json")
    parser.add_argument("--pair", choices=("qwen", "gemma", "mistral"))
    args = parser.parse_args()
    config = load_yaml("backbones.yaml")
    selected = config["pairs"]
    if args.pair:
        selected = {args.pair: selected[args.pair]}
    result = {name: audit_pair(name, pair) for name, pair in selected.items()}
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
