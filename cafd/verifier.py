"""Direct deterministic wrapper around the pinned official Manufactoria verifier."""

from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path
from typing import Any


def _official_module():
    root = Path(__file__).resolve().parents[1]
    package_root = root / "vendor" / "rl-grok-recipe"
    verifier_dir = package_root / "manufactoria" / "verifier"
    if not verifier_dir.is_dir():
        raise FileNotFoundError(
            "pinned official verifier is missing; run scripts/cafd/run_minimal.sh to materialize it"
        )
    package_name = "cafd_official_manufactoria_verifier"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(verifier_dir)]
        sys.modules[package_name] = package
    module_name = f"{package_name}.manufactoria_parser"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, verifier_dir / "manufactoria_parser.py")
        if spec is None or spec.loader is None:
            raise ImportError("cannot load official Manufactoria parser")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


_CODE_BLOCK = re.compile(r"```(?:manufactoria)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
_RESPONSE_CONTRACT = re.compile(
    r"\A```manufactoria[ \t]*\r?\n(?P<program>.*?)\r?\n```\Z",
    re.DOTALL,
)


def extract_program(completion: str) -> str:
    """Extract the first fenced Manufactoria program, or use the full response."""

    match = _CODE_BLOCK.search(completion)
    return (match.group(1) if match else completion).strip()


def extract_contract_program(completion: str) -> str | None:
    """Return the program only when the entire response obeys the fixed contract."""

    match = _RESPONSE_CONTRACT.fullmatch(completion)
    if match is None or "```" in match.group("program"):
        return None
    return match.group("program").strip()


def verify_program(program: str, test_cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Run the pinned official parser/factory and reproduce the official API score."""

    module = _official_module()
    try:
        factory = module.create_robot_factory(program)
    except Exception as exc:
        return {"valid": False, "all_passed": False, "pass_rate": 0.0, "error": str(exc), "results": []}

    results: list[dict[str, Any]] = []
    for case in test_cases:
        try:
            execution = factory.process_robot(case.get("input", ""))
            if case.get("check_output", True):
                expected_output = case.get("expected_output", "")
                has_regex = any(char in expected_output for char in [".", "+", "*", "?", "|", "(", ")"])
                if has_regex:
                    try:
                        output_matches = bool(re.fullmatch(expected_output, execution.final_tape))
                    except re.error:
                        output_matches = execution.final_tape == expected_output
                else:
                    output_matches = execution.final_tape == expected_output
                passed = (output_matches and execution.finished) == bool(case.get("expected_accepted", True))
            else:
                passed = execution.finished == bool(case.get("expected_accepted", True))
            results.append({"passed": bool(passed)})
        except Exception as exc:
            results.append({"passed": False, "error": str(exc)})
    pass_rate = sum(item["passed"] for item in results) / len(results) if results else 0.0
    return {
        "valid": True,
        "all_passed": bool(results) and all(item["passed"] for item in results),
        "pass_rate": float(pass_rate),
        "results": results,
    }


HIERARCHICAL_FORMAT_REWARD = 0.05
HIERARCHICAL_PARSE_REWARD = 0.10
HIERARCHICAL_SEMANTIC_WEIGHT = 0.90


def score_completion_details(
    completion: str,
    test_cases: list[dict[str, Any]],
    mode: str,
    *,
    require_contract: bool = True,
) -> dict[str, Any]:
    program = extract_contract_program(completion) if require_contract else extract_program(completion)
    if program is None:
        return {"reward": 0.0, "tier": "invalid_format", "pass_rate": 0.0, "valid": False}
    result = verify_program(program, test_cases)
    if mode == "hierarchical":
        if not result["valid"]:
            return {
                "reward": HIERARCHICAL_FORMAT_REWARD,
                "tier": "format_only",
                "pass_rate": 0.0,
                "valid": False,
            }
        pass_rate = float(result["pass_rate"])
        if result["all_passed"]:
            reward = 1.0
            tier = "full_pass"
        elif pass_rate > 0.0:
            reward = HIERARCHICAL_PARSE_REWARD + HIERARCHICAL_SEMANTIC_WEIGHT * pass_rate
            tier = "partial_pass"
        else:
            reward = HIERARCHICAL_PARSE_REWARD
            tier = "parse_only"
        return {"reward": reward, "tier": tier, "pass_rate": pass_rate, "valid": True}
    if mode == "pass_rate":
        reward = float(result["pass_rate"])
    elif mode == "full_pass":
        reward = float(result["all_passed"])
    else:
        raise ValueError(f"unknown Manufactoria scoring mode: {mode}")
    return {
        "reward": reward,
        "tier": "full_pass" if result["all_passed"] else ("partial_pass" if result["pass_rate"] else "parse_only"),
        "pass_rate": float(result["pass_rate"]),
        "valid": bool(result["valid"]),
    }


def score_completion(
    completion: str,
    test_cases: list[dict[str, Any]],
    mode: str,
    *,
    require_contract: bool = True,
) -> float:
    return float(
        score_completion_details(completion, test_cases, mode, require_contract=require_contract)["reward"]
    )
