"""Formal-run hard checks and pinned dependency/data materialization."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download

from .canonical import prepare as prepare_canonical
from .data import load_rows, prepare
from .layout import experiment_layout
from .modeling import (
    STUDENT_ID,
    STUDENT_REVISION,
    TEACHER_ID,
    TEACHER_REVISION,
    assert_exact_tokenizer_pair,
    load_model,
    load_tokenizers,
)
from .prompting import assert_matches_teacher_chat_template
from .training_common import write_json
from .verifier import verify_program


OFFICIAL_REPO_REVISION = "8500bec984d4a84a4aa94ca3adc31c004aa6a388"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = yaml.safe_load((root / "configs" / "cafd" / "manufactoria_has.yaml").read_text(encoding="utf-8"))
    layout = experiment_layout(root, config)
    cache = root / ".cache" / "huggingface" / "hub"
    cache.mkdir(parents=True, exist_ok=True)

    repo = root / "vendor" / "rl-grok-recipe"
    if not (repo / ".git").is_dir():
        raise RuntimeError("pinned official rl-grok-recipe checkout is missing")
    import subprocess

    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if revision != OFFICIAL_REPO_REVISION:
        raise RuntimeError(f"official verifier revision changed: {revision}")

    counts = prepare(root)
    train_rows = load_rows(root, "train")
    test_result = verify_program("START start:\n    NEXT end\n\nEND end", train_rows[0]["ground_truth"][:1])
    if "all_passed" not in test_result:
        raise RuntimeError("official deterministic verifier is not executable")

    for identifier, revision_sha in ((TEACHER_ID, TEACHER_REVISION), (STUDENT_ID, STUDENT_REVISION)):
        snapshot_download(repo_id=identifier, revision=revision_sha, cache_dir=cache)
    teacher_tokenizer, student_tokenizer = load_tokenizers(cache)
    assert_exact_tokenizer_pair(teacher_tokenizer, student_tokenizer)
    assert_matches_teacher_chat_template(teacher_tokenizer)
    generation_contract = prepare_canonical(root, student_tokenizer)
    if int(config["generation"]["max_new_tokens"]) != int(generation_contract["max_new_tokens"]):
        raise RuntimeError(
            f"configured max_new_tokens={config['generation']['max_new_tokens']} does not match "
            f"canonical bound={generation_contract['max_new_tokens']}"
        )

    # The startup contract includes actual LM-head row counts, not config-only guesses.
    teacher_model = load_model(
        TEACHER_ID,
        TEACHER_REVISION,
        cache_dir=cache,
        device="cpu",
        trainable=False,
    )
    assert_exact_tokenizer_pair(teacher_tokenizer, student_tokenizer, teacher_model=teacher_model)
    del teacher_model
    gc.collect()
    student_model = load_model(
        STUDENT_ID,
        STUDENT_REVISION,
        cache_dir=cache,
        device="cpu",
        trainable=False,
    )
    assert_exact_tokenizer_pair(teacher_tokenizer, student_tokenizer, student_model=student_model)
    del student_model
    gc.collect()

    write_json(
        layout.state_root / "hard_checks.json",
        {
            "status": "passed",
            "seed": 2027,
            "splits": counts,
            "teacher_revision": TEACHER_REVISION,
            "student_revision": STUDENT_REVISION,
            "official_repo_revision": OFFICIAL_REPO_REVISION,
            "generation_contract": generation_contract,
        },
    )
    print(json.dumps({"status": "ready", **counts}, sort_keys=True))


if __name__ == "__main__":
    main()
