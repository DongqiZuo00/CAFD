from __future__ import annotations

from typing import Any


SYSTEM_PROMPTS = {
    "code": (
        "Solve the programming problem. Think carefully, then return exactly one complete "
        "Python program in a fenced ```python code block."
    ),
    "math": (
        "Solve the problem step by step. Put only the final answer in the last "
        "\\boxed{...} expression."
    ),
}


def pair_messages(task: str, problem: str) -> list[dict[str, str]]:
    if task not in SYSTEM_PROMPTS:
        raise ValueError(f"unknown task: {task}")
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[task]},
        {"role": "user", "content": problem},
    ]


def render(tokenizer: Any, family: str, task: str, problem: str) -> str:
    messages = pair_messages(task, problem)
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if family == "qwen":
        kwargs["enable_thinking"] = True
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
        if family == "qwen" and "<think>" not in rendered:
            rendered += "<think>\n"
        return rendered

