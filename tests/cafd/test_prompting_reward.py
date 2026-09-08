from __future__ import annotations

from unittest.mock import patch

import pytest

from cafd.prompting import PROMPT_RENDERER_QWEN3_INSTRUCT, render_prompt_text
from cafd.verifier import score_completion, score_completion_details


def _fenced(program: str) -> str:
    fence = chr(96) * 3
    return f"{fence}manufactoria\n{program}\n{fence}"


def test_frozen_renderer_is_teacher_no_thinking_chat_frame() -> None:
    rendered = render_prompt_text("task", PROMPT_RENDERER_QWEN3_INSTRUCT)
    assert rendered == (
        "<|im_start|>user\n"
        "task"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert "<think>" not in rendered


def test_hierarchical_reward_has_strict_nonzero_tiers() -> None:
    tests = [{"input": "", "expected_output": ""}]
    assert score_completion("not fenced", tests, "hierarchical") == 0.0

    with patch(
        "cafd.verifier.verify_program",
        return_value={"valid": False, "all_passed": False, "pass_rate": 0.0},
    ):
        details = score_completion_details(_fenced("bad"), tests, "hierarchical")
        assert details["tier"] == "format_only"
        assert details["reward"] == pytest.approx(0.05)

    with patch(
        "cafd.verifier.verify_program",
        return_value={"valid": True, "all_passed": False, "pass_rate": 0.0},
    ):
        assert score_completion(_fenced("valid"), tests, "hierarchical") == pytest.approx(0.10)

    with patch(
        "cafd.verifier.verify_program",
        return_value={"valid": True, "all_passed": False, "pass_rate": 0.5},
    ):
        details = score_completion_details(_fenced("partial"), tests, "hierarchical")
        assert details["tier"] == "partial_pass"
        assert details["reward"] == pytest.approx(0.55)

    with patch(
        "cafd.verifier.verify_program",
        return_value={"valid": True, "all_passed": True, "pass_rate": 1.0},
    ):
        assert score_completion(_fenced("full"), tests, "hierarchical") == pytest.approx(1.0)
