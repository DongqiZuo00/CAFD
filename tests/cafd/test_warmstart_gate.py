from __future__ import annotations

from cafd.teacher_warmstart_gate import clopper_pearson_upper


def test_fixed_96_rollout_length_audit_bounds() -> None:
    upper = clopper_pearson_upper(1, 96)
    assert 0.048 < upper < 0.049
    assert clopper_pearson_upper(2, 96) > 0.05
