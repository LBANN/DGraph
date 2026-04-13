"""Shared utilities for cost-model benchmarks: timing, logging, metadata."""

import json
import os
import random
import socket
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def cuda_timed(fn: Callable, warmup: int = 10, trials: int = 50) -> list:
    """Run *fn* with CUDA-event timing. Returns per-trial wall times in seconds.

    The function is invoked with no arguments. Callers should capture any
    needed state via closure.  Warmup iterations are discarded.
    """
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(trials):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        fn()
        end_evt.record()
        torch.cuda.synchronize()
        # elapsed_time returns milliseconds
        times.append(start_evt.elapsed_time(end_evt) / 1_000.0)
    return times


# ---------------------------------------------------------------------------
# Metadata collection
# ---------------------------------------------------------------------------


def collect_metadata() -> dict:
    """Return a dict of reproducibility metadata for the current run."""
    meta: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hostname": socket.gethostname(),
    }

    # GPU info
    if torch.cuda.is_available():
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_entry = {
                "index": i,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_bytes": props.total_memory,
            }
            # UUID available in newer PyTorch builds
            if hasattr(props, "uuid"):
                gpu_entry["uuid"] = str(props.uuid)
            gpus.append(gpu_entry)
        meta["gpus"] = gpus
        meta["cuda_version"] = torch.version.cuda
    else:
        meta["gpus"] = []
        meta["cuda_version"] = None

    meta["pytorch_version"] = torch.__version__

    # NCCL version (tuple -> string)
    try:
        nccl_ver = torch.cuda.nccl.version()
        meta["nccl_version"] = ".".join(str(x) for x in nccl_ver)
    except Exception:
        meta["nccl_version"] = "unknown"

    # SLURM environment variables
    slurm_keys = [
        "SLURM_JOB_ID",
        "SLURM_NODELIST",
        "SLURM_NNODES",
        "SLURM_NTASKS",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
    ]
    meta["slurm"] = {k: os.environ.get(k) for k in slurm_keys}

    # Git commit hash of the benchmark code
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        meta["git_commit"] = result.stdout.strip()
    except Exception:
        meta["git_commit"] = "unknown"

    return meta


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def write_result(path: str, payload: dict) -> None:
    """Write *payload* as a JSON file at *path*, creating parents as needed.

    Expected schema::

        {
            "benchmark": "<name>",
            "metadata": { ... },
            "config": { ... },
            "measurements": [
                {"params": {...}, "trials_seconds": [t1, t2, ...]},
                ...
            ]
        }
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[write_result] Saved {p} ({p.stat().st_size} bytes)")


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------


def setup_distributed() -> tuple[int, int, int]:
    """Initialize torch.distributed with the env-var init method (NCCL).

    Expects MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK, and LOCAL_RANK to be
    set in the environment (standard for torchrun / SLURM + srun).

    Returns:
        (rank, world_size, local_rank)
    """
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Set Python, NumPy, and PyTorch (CPU + CUDA) random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
