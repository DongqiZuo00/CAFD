"""Pinned model loading and exact tokenizer/head compatibility checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


TEACHER_ID = "Qwen/Qwen3-4B-Instruct-2507"
TEACHER_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
STUDENT_ID = "Qwen/Qwen3-1.7B"
STUDENT_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
MODEL_VOCAB_SIZE = 151936


def load_tokenizers(cache_dir: Path) -> tuple[PreTrainedTokenizerBase, PreTrainedTokenizerBase]:
    teacher = AutoTokenizer.from_pretrained(TEACHER_ID, revision=TEACHER_REVISION, cache_dir=cache_dir)
    student = AutoTokenizer.from_pretrained(STUDENT_ID, revision=STUDENT_REVISION, cache_dir=cache_dir)
    if teacher.pad_token_id is None:
        teacher.pad_token = teacher.eos_token
    if student.pad_token_id is None:
        student.pad_token = student.eos_token
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
        if len(teacher) != len(student):
            raise AssertionError(f"vocab size differs: teacher={len(teacher)}, student={len(student)}")
        teacher_tokens = teacher.convert_ids_to_tokens(list(range(MODEL_VOCAB_SIZE)))
        student_tokens = student.convert_ids_to_tokens(list(range(MODEL_VOCAB_SIZE)))
        if teacher_tokens != student_tokens:
            first = next(i for i, pair in enumerate(zip(teacher_tokens, student_tokens)) if pair[0] != pair[1])
            raise AssertionError(
                f"token_id order differs at {first}: teacher={teacher_tokens[first]!r}, student={student_tokens[first]!r}"
            )
        if _special_ids(teacher) != _special_ids(student):
            raise AssertionError(f"special token IDs differ: {_special_ids(teacher)} != {_special_ids(student)}")
        for name, model in (("teacher", teacher_model), ("student", student_model)):
            if model is None:
                continue
            head = model.get_output_embeddings()
            config_vocab = int(model.config.get_text_config().vocab_size)
            if config_vocab != MODEL_VOCAB_SIZE or head.weight.shape[0] != MODEL_VOCAB_SIZE:
                raise AssertionError(
                    f"{name} LM-head/token-ID rows differ: config={config_vocab}, head={head.weight.shape[0]}, expected={MODEL_VOCAB_SIZE}"
                )
    except Exception as exc:
        raise RuntimeError(f"INVALID_TOKENIZER_PAIR: {exc}") from exc


def load_model(
    identifier: str,
    revision: str,
    *,
    cache_dir: Path,
    device: torch.device | str,
    trainable: bool,
) -> PreTrainedModel:
    kwargs: dict[str, Any] = {
        "cache_dir": cache_dir,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }
    if revision:
        kwargs["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(identifier, **kwargs)
    model.to(device)
    model.train(trainable)
    model.requires_grad_(trainable)
    model.config.use_cache = not trainable
    if trainable:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def base_hidden(model: PreTrainedModel, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Run only the causal backbone so full sequence×vocabulary logits never persist."""

    backbone = getattr(model, "model", None)
    if backbone is None:
        raise TypeError(f"model {type(model).__name__} does not expose a causal backbone as .model")
    output = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    return output.last_hidden_state
