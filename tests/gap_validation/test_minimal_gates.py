from gap_validation.minimal_gates import development_gates
from gap_validation.minimal_final import paired_bootstrap, wilson_interval
from gap_validation.minimal_routes import select_checkpoint

import numpy as np


def test_development_gate_requires_teacher_oracle_and_both_failures():
    gates = development_gates(t0=5, t400=12, s0=4, direct=8, oracle=12, opd=9)
    assert gates["teacher_acquisition"]
    assert gates["direct_failure"]
    assert gates["oracle_capacity"]
    assert gates["opd_failure"]
    assert gates["final_open"]


def test_direct_equal_to_teacher_falsifies_rediscovery_gap():
    gates = development_gates(t0=5, t400=12, s0=4, direct=12)
    assert not gates["direct_failure"]


def test_oracle_below_teacher_does_not_establish_capacity():
    gates = development_gates(t0=5, t400=12, s0=4, direct=8, oracle=11)
    assert not gates["oracle_capacity"]


def test_checkpoint_selection_prefers_earlier_step_on_tie(monkeypatch):
    summaries = {
        "s0": {"correct": 3, "count": 21, "exact_answer_accuracy": 3 / 21},
        "s50": {"correct": 8, "count": 21, "exact_answer_accuracy": 8 / 21},
        "s100": {"correct": 8, "count": 21, "exact_answer_accuracy": 8 / 21},
    }
    monkeypatch.setattr(
        "gap_validation.minimal_routes.summary_for_item",
        lambda item: summaries[item["condition"]],
    )
    selected = select_checkpoint(
        [
            {"condition": "s0", "checkpoint_step": 0, "model": "base"},
            {"condition": "s50", "checkpoint_step": 50, "model": "m50"},
            {"condition": "s100", "checkpoint_step": 100, "model": "m100"},
        ]
    )
    assert selected["selected_step"] == 50


def test_wilson_interval_contains_observed_fraction():
    lower, upper = wilson_interval(40, 79)
    assert lower < 40 / 79 < upper


def test_paired_bootstrap_uses_shared_items_and_positive_gap():
    values = {
        "teacher_t0": np.zeros(79),
        "teacher_t400": np.ones(79),
        "student_s0": np.zeros(79),
        "student_oracle": np.ones(79),
        "student_direct_rlvr": np.zeros(79),
        "student_final_opd": np.zeros(79),
    }
    result = paired_bootstrap(values, replicates=200)
    assert result["shared_resample_indices"]
    assert all(value > 0 for value in result["lower_bounds"].values())
