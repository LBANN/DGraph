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
# Rank placement
# ---------------------------------------------------------------------------


def gather_node_of_rank() -> list:
    """Return a list mapping each rank -> an opaque node identifier.

    Derived from the actual hostname each rank is running on, so it reflects
    real placement rather than an assumed ``rank // ranks_per_node`` layout.
    Collective: every rank must call it.
    """
    hostnames = [None] * dist.get_world_size()
    dist.all_gather_object(hostnames, socket.gethostname())
    return hostnames


def assert_placement(expected_same_node: list, context: str = "") -> None:
    """Verify the real rank->node placement matches what a benchmark assumes.

    ``expected_same_node`` is a list of (rank_a, rank_b, same) triples: *same*
    is True if the benchmark requires the two ranks to be co-located, False if
    it requires them to be on different nodes.

    Benchmarks that measure intra- vs inter-node traffic encode their layout
    assumption implicitly (in a ``--mode`` label, or a hardcoded peer map).
    Nothing about the launch command enforces it, so a wrong ``--nnodes`` /
    ``--nproc_per_node`` combination would silently mislabel intra-node
    numbers as inter-node (or vice versa) and quietly corrupt the
    hierarchical network fit that ``T_comm = max(T_intra, T_inter)`` rests on.
    Fail loudly instead.
    """
    node_of = gather_node_of_rank()
    problems = []
    for rank_a, rank_b, same in expected_same_node:
        actually_same = node_of[rank_a] == node_of[rank_b]
        if actually_same != same:
            problems.append(
                f"  ranks {rank_a} and {rank_b} must be on "
                f"{'the SAME node' if same else 'DIFFERENT nodes'}, but rank "
                f"{rank_a} is on {node_of[rank_a]!r} and rank {rank_b} is on "
                f"{node_of[rank_b]!r}"
            )
    if problems:
        raise RuntimeError(
            f"Rank placement does not match what this benchmark requires"
            f"{' (' + context + ')' if context else ''}:\n"
            + "\n".join(problems)
            + f"\n\nObserved rank -> host mapping: {list(enumerate(node_of))}\n"
            "Fix the launcher's --nnodes/--nproc_per_node (torchrun) or "
            "-N/--ntasks-per-node (srun) so the layout matches."
        )


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Set Python, NumPy, and PyTorch (CPU + CUDA) random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
