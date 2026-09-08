from __future__ import annotations

from typing import Any


def development_gates(
    *, t0: int, t400: int, s0: int, direct: int | None = None,
    oracle: int | None = None, opd: int | None = None,
) -> dict[str, Any]:
    values = {"t0": t0, "t400": t400, "s0": s0, "direct": direct, "oracle": oracle, "opd": opd}
    teacher_acquisition = t400 > t0
    direct_failure = direct is not None and direct < t400
    oracle_capacity = oracle is not None and oracle > s0 and oracle >= t400
    opd_failure = (
        opd is not None
        and oracle is not None
        and direct is not None
        and opd < min(t400, oracle)
    )
    final_open = bool(
        teacher_acquisition
        and oracle is not None
        and direct is not None
        and opd is not None
        and oracle > s0
        and direct < min(t400, oracle)
        and opd < min(t400, oracle)
    )
    return {
        "values": values,
        "teacher_acquisition": teacher_acquisition,
        "direct_failure": direct_failure,
        "oracle_capacity": oracle_capacity,
        "opd_failure": opd_failure,
        "final_open": final_open,
    }
