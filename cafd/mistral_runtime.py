"""Pinned Mistral-only runtime for isolated CAFD experiments."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Mistral3ForConditionalGeneration,
    Ministral3ForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .prompting import PROMPT_RENDERER_MISTRAL3_INSTRUCT


TEACHER_ID = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
TEACHER_REVISION = "f6fae9795746f63c9be8344932f01275f3c63734"
STUDENT_ID = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
STUDENT_REVISION = "b6d637bef2393152b3da2b2fde72eecdee30557e"
MODEL_VOCAB_SIZE = 131072
PROTOCOL = "mistral_cafd_only_v6"
RUN_ID = "mistral_cafd_only_v6"
V7_PROTOCOL = "mistral_cafd_disjoint_v7"
V7_RUN_ID = "mistral_cafd_disjoint_v7"
ALLOWED_IDENTITIES = {
    (PROTOCOL, RUN_ID),
    (V7_PROTOCOL, V7_RUN_ID),
}


def validate_config(config: dict[str, Any]) -> None:
    identity = (str(config["protocol"]), str(config["experiment"]["run_id"]))
    if identity not in ALLOWED_IDENTITIES:
        raise RuntimeError(f"wrong Mistral CAFD identity: {identity}")
    expected = {
        "teacher": (TEACHER_ID, TEACHER_REVISION),
        "student": (STUDENT_ID, STUDENT_REVISION),
    }
    for role, (identifier, revision) in expected.items():
        record = config["models"][role]
        actual = (str(record["id"]), str(record["revision"]))
        if actual != (identifier, revision):
            raise RuntimeError(f"unpinned {role} model: {actual}")
        if "qwen" in actual[0].lower():
            raise RuntimeError(f"QWEN_BANNED: {actual[0]}")
    if config["generation"]["prompt_renderer"] != PROMPT_RENDERER_MISTRAL3_INSTRUCT:
        raise RuntimeError("Mistral run requires the frozen Mistral renderer")


def load_tokenizers(
    cache_dir: Path,
) -> tuple[PreTrainedTokenizerBase, PreTrainedTokenizerBase]:
    common = {"cache_dir": cache_dir, "fix_mistral_regex": True}
    teacher = AutoTokenizer.from_pretrained(
        TEACHER_ID, revision=TEACHER_REVISION, **common
    )
    student = AutoTokenizer.from_pretrained(
        STUDENT_ID, revision=STUDENT_REVISION, **common
    )
    return teacher, student


def _special_ids(tokenizer: PreTrainedTokenizerBase) -> dict[str, Any]:
    return {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "all_special_ids": list(tokenizer.all_special_ids),
    }


def assert_exact_tokenizer_pair(
    teacher: PreTrainedTokenizerBase,
    student: PreTrainedTokenizerBase,
    teacher_model: PreTrainedModel | None = None,
    student_model: PreTrainedModel | None = None,
) -> None:
    try:
        if len(teacher) != MODEL_VOCAB_SIZE or len(student) != MODEL_VOCAB_SIZE:
            raise AssertionError(
                f"vocabulary size differs from {MODEL_VOCAB_SIZE}: "
                f"teacher={len(teacher)}, student={len(student)}"
            )
        teacher_tokens = teacher.convert_ids_to_tokens(range(MODEL_VOCAB_SIZE))
        student_tokens = student.convert_ids_to_tokens(range(MODEL_VOCAB_SIZE))
        if teacher_tokens != student_tokens:
            first = next(
                index
                for index, pair in enumerate(zip(teacher_tokens, student_tokens))
                if pair[0] != pair[1]
            )
            raise AssertionError(
                f"token-ID order differs at {first}: "
                f"{teacher_tokens[first]!r} != {student_tokens[first]!r}"
            )
        if _special_ids(teacher) != _special_ids(student):
            raise AssertionError("Mistral special-token IDs differ")
        for name, model in (("teacher", teacher_model), ("student", student_model)):
            if model is None:
                continue
            head = model.get_output_embeddings()
            config_vocab = int(model.config.get_text_config().vocab_size)
            if config_vocab != MODEL_VOCAB_SIZE or head.weight.shape[0] != MODEL_VOCAB_SIZE:
                raise AssertionError(
                    f"{name} LM head mismatch: config={config_vocab}, "
                    f"head={head.weight.shape[0]}"
                )
    except Exception as exc:
        raise RuntimeError(f"INVALID_MISTRAL_TOKENIZER_PAIR: {exc}") from exc


def _text_only_mistral3(
    identifier: str,
    kwargs: dict[str, Any],
) -> PreTrainedModel:
    full = Mistral3ForConditionalGeneration.from_pretrained(identifier, **kwargs)
    text = Ministral3ForCausalLM(full.config.text_config)
    text.model = full.model.language_model
    text.lm_head = full.lm_head
    text.generation_config = full.generation_config
    del full
    return text


def load_model(
    identifier: str,
    revision: str,
    *,
    cache_dir: Path,
    device: torch.device | str,
    trainable: bool,
) -> PreTrainedModel:
    if "qwen" in identifier.lower():
        raise RuntimeError(f"QWEN_BANNED: refusing model/checkpoint {identifier}")
    kwargs: dict[str, Any] = {
        "cache_dir": cache_dir,
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }
    if revision:
        kwargs["revision"] = revision
    config = AutoConfig.from_pretrained(
        identifier,
        cache_dir=cache_dir,
        revision=revision or "main",
    )
    if config.model_type == "mistral3":
        model = _text_only_mistral3(identifier, kwargs)
    elif config.model_type == "ministral3":
        model = AutoModelForCausalLM.from_pretrained(identifier, **kwargs)
    else:
        raise RuntimeError(
            f"MISTRAL_ONLY: unsupported model_type={config.model_type!r} at {identifier}"
        )
    model.to(device)
    model.train(trainable)
    model.requires_grad_(trainable)
    model.config.use_cache = not trainable
    if trainable:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model


def install_into(module: ModuleType) -> None:
    """Patch a legacy trainer module only inside a Mistral entrypoint process."""

    module.load_tokenizers = load_tokenizers
    module.load_model = load_model
    module.assert_exact_tokenizer_pair = assert_exact_tokenizer_pair
    if hasattr(module, "TEACHER_ID"):
        module.TEACHER_ID = TEACHER_ID
    if hasattr(module, "TEACHER_REVISION"):
        module.TEACHER_REVISION = TEACHER_REVISION
    if hasattr(module, "STUDENT_ID"):
        module.STUDENT_ID = STUDENT_ID
    if hasattr(module, "STUDENT_REVISION"):
        module.STUDENT_REVISION = STUDENT_REVISION
