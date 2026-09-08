from __future__ import annotations

from gap_validation.splits import stable_split, stratified_holdout


def test_stable_split_is_input_order_independent():
    rows = [{"id": str(index)} for index in range(100)]
    a_train, a_validation = stable_split(rows, "id", 42, 0.1)
    b_train, b_validation = stable_split(reversed(rows), "id", 42, 0.1)
    assert {row["id"] for row in a_train} == {row["id"] for row in b_train}
    assert {row["id"] for row in a_validation} == {row["id"] for row in b_validation}
    assert {row["id"] for row in a_train}.isdisjoint({row["id"] for row in a_validation})


def test_stratified_holdout_preserves_disjoint_partition():
    rows = [
        {"id": f"{subject}-{index}", "subject": subject}
        for subject in ("algebra", "geometry", "number_theory")
        for index in range(10)
    ]
    development, test = stratified_holdout(rows, "id", "subject", 42, 0.2)
    assert len(development) == 6
    assert len(test) == 24
    assert {row["id"] for row in development}.isdisjoint({row["id"] for row in test})
    assert {row["subject"] for row in development} == {"algebra", "geometry", "number_theory"}
