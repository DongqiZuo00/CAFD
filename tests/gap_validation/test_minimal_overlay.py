from gap_validation.minimal_prepare_oracle import canonical_completion, prepare_oracle_rows
from gap_validation.minimal_runtime import completion_tokens_from_history, generation_events_from_history


def test_canonical_oracle_rows_are_verified_and_repeated_deterministically():
    rows = [
        {"id": "a", "prompt": "one", "solution": "42", "stream_position": 0},
        {"id": "b", "prompt": "two", "solution": r"\frac{1}{2}", "stream_position": 1},
    ]
    exposures, failures = prepare_oracle_rows(rows, seed=42, exposure_count=5)
    assert not failures
    assert len(exposures) == 5
    assert all(row["verifier_output"]["correct"] for row in exposures)
    assert {row["problem_id"] for row in exposures} == {"a", "b"}


def test_canonical_completion_preserves_existing_box():
    assert canonical_completion(r"\boxed{3}") == r"\boxed{3}"
    assert canonical_completion("3") == r"\boxed{3}"


def test_completion_token_accounting_uses_all_logged_updates():
    history = [
        {"completions/mean_length": 10.0},
        {"loss": 1.0},
        {"completions/mean_length": 12.5},
    ]
    assert completion_tokens_from_history(history, slots_per_update=32) == 720
    assert generation_events_from_history(history) == 2
