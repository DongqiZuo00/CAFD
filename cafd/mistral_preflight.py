"""No-test-data preflight and Mistral-tokenized canonical materialization."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from .canonical import prepare
from .configuration import load_experiment_config
from .layout import experiment_layout
from .mistral_runtime import (
    STUDENT_ID,
    STUDENT_REVISION,
    TEACHER_ID,
    TEACHER_REVISION,
    assert_exact_tokenizer_pair,
    load_tokenizers,
    validate_config,
)
from .prompting import assert_matches_mistral_chat_template
from .training_common import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = load_experiment_config(root)
    validate_config(config)
    cache = root / ".cache" / "huggingface" / "hub"
    cache.mkdir(parents=True, exist_ok=True)

    for identifier, revision in (
        (TEACHER_ID, TEACHER_REVISION),
        (STUDENT_ID, STUDENT_REVISION),
    ):
        snapshot_download(
            repo_id=identifier,
            revision=revision,
            cache_dir=cache,
            allow_patterns=[
                "*.json",
                "*.jinja",
                "*.txt",
                "model-*.safetensors",
                "tokenizer*",
                "special_tokens_map.json",
            ],
        )

    teacher, student = load_tokenizers(cache)
    assert_exact_tokenizer_pair(teacher, student)
    assert_matches_mistral_chat_template(teacher)
    assert_matches_mistral_chat_template(student)
    contract = prepare(root, student)
    if int(contract["max_new_tokens"]) != int(config["generation"]["max_new_tokens"]):
        raise RuntimeError(
            f"generation bound changed: canonical={contract['max_new_tokens']} "
            f"config={config['generation']['max_new_tokens']}"
        )
    if int(contract["verified_training_solutions"]) != 678:
        raise RuntimeError(f"canonical training count changed: {contract}")

    layout = experiment_layout(root, config)
    write_json(
        layout.state_root / "mistral_preflight.json",
        {
            "status": "passed",
            "qwen_banned": True,
            "teacher": {"id": TEACHER_ID, "revision": TEACHER_REVISION},
            "student": {"id": STUDENT_ID, "revision": STUDENT_REVISION},
            "vocab_size": len(student),
            "token_id_mismatches": 0,
            "renderer": config["generation"]["prompt_renderer"],
            "canonical": contract,
            "frozen_test_accessed": False,
        },
    )


if __name__ == "__main__":
    main()
