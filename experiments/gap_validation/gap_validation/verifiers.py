from __future__ import annotations

import os
import re
import resource
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


@dataclass(frozen=True)
class VerificationResult:
    correct: bool
    status: str
    stdout: str = ""
    stderr: str = ""
    runtime_seconds: float | None = None
    parsed_answer: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_python(text: str) -> str | None:
    matches = re.findall(r"```python\s*\n(.*?)\n```", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return max(matches, key=len).strip()
    plain = re.findall(r"```\s*\n(.*?)\n```", text, flags=re.DOTALL)
    return max(plain, key=len).strip() if plain else None


def _limit_process(memory_bytes: int, cpu_seconds: int) -> None:
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    os.setsid()


def verify_stdio_python(
    generated: str,
    test_cases: list[dict[str, Any]],
    timeout_seconds: float = 5.0,
    memory_bytes: int = 1024**3,
) -> VerificationResult:
    code = extract_python(generated)
    if not code:
        return VerificationResult(False, "MALFORMED_OUTPUT")
    with tempfile.TemporaryDirectory(prefix="gap-verify-") as directory:
        program = Path(directory) / "submission.py"
        program.write_text(code, encoding="utf-8")
        for case in test_cases:
            if case.get("type") != "stdin_stdout":
                return VerificationResult(False, "UNSUPPORTED_TEST_TYPE")
            try:
                completed = subprocess.run(
                    ["python", "-I", str(program)],
                    input=case.get("input", ""),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout_seconds,
                    cwd=directory,
                    env={"PATH": os.environ.get("PATH", "")},
                    preexec_fn=lambda: _limit_process(memory_bytes, max(1, int(timeout_seconds))),
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                return VerificationResult(False, "TIMEOUT", stdout=error.stdout or "", stderr=error.stderr or "")
            if completed.returncode != 0:
                status = "SYNTAX_OR_COMPILE_ERROR" if "SyntaxError" in completed.stderr else "RUNTIME_ERROR"
                return VerificationResult(False, status, completed.stdout, completed.stderr)
            actual = "\n".join(line.rstrip() for line in completed.stdout.strip().splitlines())
            expected = "\n".join(line.rstrip() for line in str(case.get("output", "")).strip().splitlines())
            if actual != expected:
                return VerificationResult(False, "WRONG_ANSWER", completed.stdout, completed.stderr)
        return VerificationResult(True, "ACCEPTED")


def extract_boxed(text: str) -> str | None:
    positions = [match.start() for match in re.finditer(r"(?<!\\)\\boxed\{", text)]
    if not positions:
        return None
    start = positions[-1] + len("\\boxed{")
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
        elif text[index] == "}" and (index == 0 or text[index - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[start:index]
    return None


def verify_math(generated: str, gold: str) -> VerificationResult:
    extracted = extract_boxed(generated)
    if extracted is None:
        return VerificationResult(False, "MALFORMED_OUTPUT")
    config = LatexExtractionConfig(
        normalization_config=NormalizationConfig(
            nits=False,
            malformed_operators=False,
            basic_latex=True,
            equations=True,
            boxed="all",
            units=True,
        ),
        boxed_match_priority=0,
        try_extract_without_anchor=False,
    )
    try:
        answer_parsed = parse(f"\\boxed{{{extracted}}}", extraction_config=[config], extraction_mode="first_match")
        gold_parsed = parse(str(gold), extraction_mode="first_match")
        correct = bool(verify(gold_parsed, answer_parsed))
    except Exception as error:
        return VerificationResult(False, "PARSER_ERROR", stderr=repr(error), parsed_answer=extracted)
    return VerificationResult(correct, "ACCEPTED" if correct else "WRONG_ANSWER", parsed_answer=extracted)

