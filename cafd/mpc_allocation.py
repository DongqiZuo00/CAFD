"""Fail-closed resource guard for the single-GPU CAFD-MPC integration runner."""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

MAX_MEMORY_MIB = 192 * 1024
MAX_CAFD_GPUS = 2


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)):
        raise RuntimeError(f"invalid {name}: {value!r}")
    number = int(value)
    if number < 1:
        raise RuntimeError(f"{name} must be positive")
    return number


def _memory_mib(value: str, name: str) -> float:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([KMGT](?:I?B)?|B)?", str(value).strip(), re.I)
    if match is None:
        raise RuntimeError(f"invalid Slurm memory metadata {name}={value!r}")
    number = float(match[1])
    suffix = (match[2] or "").upper()
    unit = suffix[:1] if suffix else "M"
    factors = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024, "B": 1 / (1024 * 1024)}
    mib = number * factors[unit]
    if not math.isfinite(mib) or mib <= 0:
        raise RuntimeError(f"Slurm memory reservation is unknown/unbounded: {name}={value!r}")
    return mib


def _cpus_on_node(environ: Mapping[str, str]) -> int:
    if environ.get("SLURM_CPUS_ON_NODE"):
        return _integer(environ["SLURM_CPUS_ON_NODE"], "SLURM_CPUS_ON_NODE")
    value = environ.get("SLURM_JOB_CPUS_PER_NODE", "")
    match = re.fullmatch(r"([0-9]+)(?:\(x1\))?", value)
    if match is None:
        raise RuntimeError("per-CPU memory needs unambiguous allocated CPUs on one node")
    return _integer(match[1], "SLURM_JOB_CPUS_PER_NODE")


def validate_allocation(environ: Mapping[str, str], gpu_name: str, visible_gpu_count: int) -> dict[str, Any]:
    """Validate the actual Slurm allocation, not just requested configuration.

    Bare Slurm memory values are MiB. Explicit K/M/G/T suffixes are accepted.
    SLURM_CPUS_PER_TASK is deliberately insufficient for per-node memory: an
    allocation may contain several tasks, so only allocated CPU counts qualify.
    """
    job_id = str(_integer(environ.get("SLURM_JOB_ID", ""), "SLURM_JOB_ID"))
    if isinstance(visible_gpu_count, bool) or visible_gpu_count != 1:
        raise RuntimeError("CAFD-MPC runner requires exactly one visible GPU")
    if not re.search(r"\bB200\b", str(gpu_name), re.I):
        raise RuntimeError(f"CAFD-MPC requires NVIDIA B200, got {gpu_name!r}")
    for key in ("SLURM_JOB_NUM_NODES", "SLURM_NNODES"):
        if environ.get(key) and _integer(environ[key], key) != 1:
            raise RuntimeError("CAFD-MPC runner requires a single-node allocation")
    gpu_counts = {}
    for key in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE"):
        if environ.get(key):
            gpu_counts[key] = _integer(environ[key], key)
            if gpu_counts[key] > MAX_CAFD_GPUS:
                raise RuntimeError(f"Slurm reservation exceeds {MAX_CAFD_GPUS} GPUs")
    if environ.get("SLURM_JOB_GPUS"):
        ids = environ["SLURM_JOB_GPUS"].split(",")
        if any(not item.strip() for item in ids):
            raise RuntimeError("invalid SLURM_JOB_GPUS metadata")
        gpu_counts["SLURM_JOB_GPUS"] = len(ids)
        if len(ids) > MAX_CAFD_GPUS:
            raise RuntimeError(f"Slurm reservation exceeds {MAX_CAFD_GPUS} GPUs")
    memory = []
    cpus = None
    if environ.get("SLURM_MEM_PER_NODE"):
        memory.append(("SLURM_MEM_PER_NODE", _memory_mib(environ["SLURM_MEM_PER_NODE"], "SLURM_MEM_PER_NODE")))
    if environ.get("SLURM_MEM_PER_CPU"):
        cpus = _cpus_on_node(environ)
        memory.append(("SLURM_MEM_PER_CPU*allocated_cpus",
                       _memory_mib(environ["SLURM_MEM_PER_CPU"], "SLURM_MEM_PER_CPU") * cpus))
    if not memory:
        raise RuntimeError("missing Slurm memory reservation; refusing unknown memory allocation")
    reserved = max(value for _, value in memory)
    if reserved > MAX_MEMORY_MIB:
        raise RuntimeError(f"Slurm memory reservation {reserved:g} MiB exceeds {MAX_MEMORY_MIB} MiB")
    return {
        "job_id": job_id, "gpu_name": str(gpu_name), "visible_gpu_count": 1,
        "maximum_cafd_gpus": MAX_CAFD_GPUS, "slurm_gpu_count_metadata": gpu_counts,
        "reserved_memory_mib": reserved, "maximum_memory_mib": MAX_MEMORY_MIB,
        "allocated_cpus_for_memory": cpus, "memory_basis": [name for name, _ in memory],
        "single_gpu_runner": True,
    }
