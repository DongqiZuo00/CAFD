from __future__ import annotations

import sys

import pytest


pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the production sandbox uses Linux rlimits")

from gap_validation.verifiers import extract_boxed, verify_math, verify_stdio_python  # noqa: E402


CASES = [{"type": "stdin_stdout", "input": "2 3\n", "output": "5\n"}]


def fenced(program: str) -> str:
    return f"```python\n{program}\n```"


def test_code_correct():
    assert verify_stdio_python(fenced("a,b=map(int,input().split());print(a+b)"), CASES).status == "ACCEPTED"


def test_code_wrong_answer():
    assert verify_stdio_python(fenced("print(0)"), CASES).status == "WRONG_ANSWER"


def test_code_syntax_error():
    assert verify_stdio_python(fenced("if True print(1)"), CASES).status == "SYNTAX_OR_COMPILE_ERROR"


def test_code_runtime_error():
    assert verify_stdio_python(fenced("raise RuntimeError('x')"), CASES).status == "RUNTIME_ERROR"


def test_code_timeout():
    result = verify_stdio_python(fenced("while True: pass"), CASES, timeout_seconds=0.2)
    assert result.status == "TIMEOUT"


def test_code_malformed_output():
    assert verify_stdio_python("there is no fenced program", CASES).status == "MALFORMED_OUTPUT"


def test_boxed_parser_uses_last_balanced_answer():
    assert extract_boxed(r"scratch \boxed{1}; final \boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_math_correct_numerical():
    assert verify_math(r"Therefore \boxed{42}", r"42").correct


def test_math_wrong_numerical():
    assert verify_math(r"Therefore \boxed{41}", r"42").status == "WRONG_ANSWER"


def test_math_symbolic_equivalence():
    assert verify_math(r"Therefore \boxed{\frac{1}{2}}", r"0.5").correct


def test_math_malformed():
    assert verify_math("answer is 42", "42").status == "MALFORMED_OUTPUT"
