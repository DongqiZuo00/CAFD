from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import psutil
import torch
import torch.distributed as dist

from .io import ARTIFACT_ROOT, atomic_json


def command(args: list[str]) -> dict[str, Any]:
    executable = shutil.which(args[0])
    if executable is None:
        return {"command": args, "returncode": None, "stdout": "", "stderr": "not found"}
    completed = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {
        "command": args,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def probe() -> dict[str, Any]:
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "capability": list(torch.cuda.get_device_capability(index)),
                "multi_processor_count": properties.multi_processor_count,
            }
        )
    payload: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version() if torch.cuda.is_available() else None,
        "host_memory": dict(psutil.virtual_memory()._asdict()),
        "devices": devices,
        "nvidia_smi_query": command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version,pci.bus_id",
                "--format=csv,noheader,nounits",
            ]
        ),
        "topology": command(["nvidia-smi", "topo", "-m"]),
        "local_storage": command(["df", "-h", "/tmp", "/blue/du.j/jinjiaguo/CAFD"]),
        "block_devices": command(["lsblk", "-o", "NAME,TYPE,SIZE,MODEL,MOUNTPOINTS"]),
        "attention": {
            "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled() if torch.cuda.is_available() else False,
            "flash_attention_2_importable": False,
        },
    }
    try:
        import flash_attn  # type: ignore

        payload["attention"]["flash_attention_2_importable"] = True
        payload["attention"]["flash_attn_version"] = flash_attn.__version__
    except Exception as error:
        payload["attention"]["flash_attn_error"] = repr(error)
    return payload


def distributed_all_reduce() -> dict[str, Any]:
    if "RANK" not in os.environ:
        return {"status": "SKIPPED", "reason": "not launched with torchrun"}
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    value = torch.tensor([rank + 1.0], device="cuda")
    dist.all_reduce(value)
    expected = world * (world + 1) / 2
    torch.cuda.synchronize()
    result = {
        "rank": rank,
        "world_size": world,
        "observed": value.item(),
        "expected": expected,
        "passed": value.item() == expected,
    }
    dist.barrier()
    dist.destroy_process_group()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_ROOT / "hardware_probe")
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {"hardware": probe(), "all_reduce": distributed_all_reduce()}
    atomic_json(args.output_dir / f"rank-{rank}.json", result)
    if rank == 0:
        atomic_json(ARTIFACT_ROOT / "hardware_topology.json", result)
        markdown = ["# Hardware topology", "", f"- Host: `{result['hardware']['hostname']}`"]
        for device in result["hardware"]["devices"]:
            gib = device["total_memory_bytes"] / 1024**3
            markdown.append(
                f"- GPU {device['index']}: `{device['name']}`, {gib:.2f} GiB, capability {device['capability']}"
            )
        markdown.extend(["", "## NVIDIA topology", "", "```", result["hardware"]["topology"]["stdout"], "```"])
        (ARTIFACT_ROOT / "hardware_topology.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

