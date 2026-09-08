from __future__ import annotations

from cafd.evaluate_condition import _final_record


def test_final_record_internal_condition_cannot_be_overwritten() -> None:
    frozen = {
        "condition": "Direct RLVR",
        "label": "Direct RLVR",
        "checkpoint": "/checkpoint",
        "generated_tokens": 1234,
    }
    stats = {
        "condition": "another accidental value",
        "correct": 17,
        "total": 132,
        "accuracy": 17 / 132,
        "generated_tokens": 5678,
    }

    record = _final_record("direct", frozen, stats)

    assert record["condition"] == "direct"
    assert record["label"] == "Direct RLVR"
    assert record["correct"] == 17
    assert record["generated_tokens"] == 1234
    assert record["evaluation_generated_tokens"] == 5678
