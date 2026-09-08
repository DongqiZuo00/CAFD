"""One frozen prompt renderer shared by Teacher, Student, and capacity control."""

from __future__ import annotations

from typing import Any

from transformers import PreTrainedTokenizerBase


PROMPT_RENDERER_RAW = "raw_plain_text"
PROMPT_RENDERER_QWEN3_INSTRUCT = "qwen3_instruct_no_thinking"
PROMPT_RENDERER_MISTRAL3_INSTRUCT = "ministral3_instruct_fixed_system"
MISTRAL_SYSTEM_PROMPT = "Return exactly one Manufactoria DSL code block and nothing else."


def render_prompt_text(prompt: str, renderer: str) -> str:
    if renderer == PROMPT_RENDERER_RAW:
        return prompt
    if renderer == PROMPT_RENDERER_QWEN3_INSTRUCT:
        return (
            "<|im_start|>user\n"
            + prompt
            + "<|im_end|>\n"
            + "<|im_start|>assistant\n"
        )
    if renderer == PROMPT_RENDERER_MISTRAL3_INSTRUCT:
        return (
            "<s>[SYSTEM_PROMPT]"
            + MISTRAL_SYSTEM_PROMPT
            + "[/SYSTEM_PROMPT][INST]"
            + prompt
            + "[/INST]"
        )
    raise ValueError(f"unknown prompt renderer: {renderer}")


def encode_prompt(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    renderer: str,
    max_length: int | None = None,
    return_tensors: str | None = None,
) -> Any:
    rendered = render_prompt_text(prompt, renderer)
    kwargs: dict[str, Any] = {"add_special_tokens": renderer == PROMPT_RENDERER_RAW}
    if max_length is not None:
        kwargs.update(truncation=True, max_length=max_length)
    if return_tensors is not None:
        kwargs["return_tensors"] = return_tensors
    return tokenizer(rendered, **kwargs)


def prompt_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    renderer: str,
    max_length: int | None = None,
) -> list[int]:
    encoded = encode_prompt(tokenizer, prompt, renderer=renderer, max_length=max_length)
    return list(encoded["input_ids"])


def assert_matches_teacher_chat_template(tokenizer: PreTrainedTokenizerBase) -> None:
    probe = "renderer-contract-probe"
    official = tokenizer.apply_chat_template(
        [{"role": "user", "content": probe}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    frozen = render_prompt_text(probe, PROMPT_RENDERER_QWEN3_INSTRUCT)
    if official != frozen:
        raise RuntimeError("frozen prompt renderer no longer matches the pinned Teacher chat template")
    official_ids = tokenizer(official, add_special_tokens=False)["input_ids"]
    frozen_ids = prompt_token_ids(tokenizer, probe, renderer=PROMPT_RENDERER_QWEN3_INSTRUCT)
    if list(official_ids) != frozen_ids:
        raise RuntimeError("frozen prompt renderer changed Teacher prompt token IDs")


def assert_matches_mistral_chat_template(
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    probe = "renderer-contract-probe"
    official = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": MISTRAL_SYSTEM_PROMPT},
            {"role": "user", "content": probe},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    frozen = render_prompt_text(probe, PROMPT_RENDERER_MISTRAL3_INSTRUCT)
    if official != frozen:
        raise RuntimeError(
            "frozen Mistral renderer no longer matches the pinned chat template"
        )
    official_ids = tokenizer(official, add_special_tokens=False)["input_ids"]
    frozen_ids = prompt_token_ids(
        tokenizer, probe, renderer=PROMPT_RENDERER_MISTRAL3_INSTRUCT
    )
    if list(official_ids) != frozen_ids:
        raise RuntimeError("frozen Mistral renderer changed prompt token IDs")
