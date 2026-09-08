import pytest

from cafd.mpc_allocation import MAX_MEMORY_MIB, validate_allocation


def valid(**kwargs):
    return {"SLURM_JOB_ID": "41234567", "SLURM_MEM_PER_NODE": "196608",
            "SLURM_GPUS_ON_NODE": "1", "SLURM_JOB_NUM_NODES": "1", **kwargs}


def test_single_b200_192g_boundary_passes():
    result = validate_allocation(valid(), "NVIDIA B200", 1)
    assert result["reserved_memory_mib"] == MAX_MEMORY_MIB
    assert result["maximum_cafd_gpus"] == 2


@pytest.mark.parametrize("value,expected", [("96G", 98304), ("192GiB", 196608),
    ("192GB", 196608), ("1024K", 1), ("64", 64), ("0.1875T", 196608)])
def test_explicit_memory_units(value, expected):
    result = validate_allocation(valid(SLURM_MEM_PER_NODE=value), "NVIDIA B200", 1)
    assert result["reserved_memory_mib"] == expected


@pytest.mark.parametrize("extra", [
    {"SLURM_JOB_ID": ""}, {"SLURM_JOB_ID": "abc"}, {"SLURM_JOB_ID": "0"},
    {"SLURM_MEM_PER_NODE": ""}, {"SLURM_MEM_PER_NODE": "0"},
    {"SLURM_MEM_PER_NODE": "all"}, {"SLURM_MEM_PER_NODE": "196609"},
    {"SLURM_MEM_PER_NODE": "193G"}, {"SLURM_MEM_PER_NODE": "nan"},
    {"SLURM_MEM_PER_NODE": "-1"}, {"SLURM_JOB_NUM_NODES": "2"},
    {"SLURM_GPUS_ON_NODE": "3"}, {"SLURM_GPUS": "3"},
    {"SLURM_JOB_GPUS": "0,1,2"},
])
def test_invalid_or_overbudget_allocation_fails(extra):
    with pytest.raises(RuntimeError):
        validate_allocation(valid(**extra), "NVIDIA B200", 1)


@pytest.mark.parametrize("name,count", [("NVIDIA A100", 1), ("NVIDIA B200", 0),
                                      ("NVIDIA B200", 2), ("B200", True)])
def test_wrong_device_or_visible_count_fails(name, count):
    with pytest.raises(RuntimeError):
        validate_allocation(valid(), name, count)


def test_per_cpu_memory_uses_allocated_cpus_not_cpus_per_task():
    env = valid(SLURM_MEM_PER_NODE="", SLURM_MEM_PER_CPU="8192", SLURM_CPUS_ON_NODE="24")
    result = validate_allocation(env, "NVIDIA B200", 1)
    assert result["reserved_memory_mib"] == 196608
    assert result["allocated_cpus_for_memory"] == 24
    env["SLURM_CPUS_ON_NODE"] = "25"
    with pytest.raises(RuntimeError, match="exceeds"):
        validate_allocation(env, "NVIDIA B200", 1)


def test_per_cpu_memory_single_node_cpu_list_supported():
    env = valid(SLURM_MEM_PER_NODE="", SLURM_MEM_PER_CPU="4G",
                SLURM_JOB_CPUS_PER_NODE="24(x1)")
    assert validate_allocation(env, "NVIDIA B200", 1)["reserved_memory_mib"] == 98304


@pytest.mark.parametrize("cpus", ["", "24(x2)", "24,24", "wrong"])
def test_ambiguous_per_cpu_metadata_fails(cpus):
    env = valid(SLURM_MEM_PER_NODE="", SLURM_MEM_PER_CPU="4G",
                SLURM_JOB_CPUS_PER_NODE=cpus, SLURM_CPUS_PER_TASK="1")
    with pytest.raises(RuntimeError):
        validate_allocation(env, "NVIDIA B200", 1)


def test_conflicting_memory_metadata_cannot_hide_overbudget_reservation():
    env = valid(SLURM_MEM_PER_CPU="16G", SLURM_CPUS_ON_NODE="16")
    with pytest.raises(RuntimeError, match="exceeds"):
        validate_allocation(env, "NVIDIA B200", 1)
