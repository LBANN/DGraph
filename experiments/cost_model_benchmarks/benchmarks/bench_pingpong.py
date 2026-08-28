"""Benchmark 1.1 — Network Bandwidth / Latency (Ping-Pong).

Measures one-way transfer time across a sweep of message sizes using a
two-rank ping-pong pattern.  Run once with both ranks on the same node
(intra-node, NVLink) and once with ranks on different nodes (inter-node,
InfiniBand).  The SLURM script controls placement; this script only records
a --mode label.

Usage (via torchrun / srun)::

    torchrun --nnodes 1 --nproc_per_node 2 \\
        -m benchmarks.bench_pingpong \\
        --mode intra --min-bytes 64 --max-bytes 67108864 --steps 21 \\
        --warmup 20 --trials 100 --output data/pingpong_intra.json --seed 42
"""

import argparse
import os

import torch
import torch.distributed as dist

from benchmarks.common import (
    assert_placement,
    collect_metadata,
    seed_everything,
    setup_distributed,
    write_result,
)


# ---------------------------------------------------------------------------
# Ping-pong timing
# ---------------------------------------------------------------------------

def pingpong_timed(rank: int, tensor: torch.Tensor, warmup: int, trials: int) -> list:
    """Perform a ping-pong between rank 0 and rank 1.

    Returns per-trial *one-way* transfer times in seconds (rank 0 only).
    Rank 1 returns an empty list.
    """
    # Warmup
    for _ in range(warmup):
        if rank == 0:
            dist.send(tensor, dst=1)
            dist.recv(tensor, src=1)
        else:
            dist.recv(tensor, src=0)
            dist.send(tensor, dst=0)
    torch.cuda.synchronize()
    dist.barrier()

    times = []
    for _ in range(trials):
        dist.barrier()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        if rank == 0:
            start_evt.record()
            dist.send(tensor, dst=1)
            dist.recv(tensor, src=1)
            end_evt.record()
            torch.cuda.synchronize()
            # Round-trip / 2 = one-way
            times.append(start_evt.elapsed_time(end_evt) / 2.0 / 1_000.0)
        else:
            dist.recv(tensor, src=0)
            dist.send(tensor, dst=0)

    return times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Ping-pong bandwidth/latency benchmark")
    p.add_argument("--min-bytes", type=int, default=64)
    p.add_argument("--max-bytes", type=int, default=67_108_864)  # 64 MiB
    p.add_argument("--steps", type=int, default=21,
                   help="Number of logarithmically-spaced message sizes")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--mode", choices=["intra", "inter"], default="inter",
                   help="Label only — actual placement is controlled by SLURM")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()

    if world_size != 2:
        raise ValueError(f"bench_pingpong requires exactly 2 ranks, got {world_size}")

    # --mode is only a label written into the output JSON; it does not change
    # which ranks talk to each other (always 0 <-> 1). Verify the launcher
    # actually placed those two ranks the way the label claims, so an
    # --nnodes/--nproc_per_node mistake can't silently produce intra-node
    # timings filed as inter-node (or vice versa).
    assert_placement(
        [(0, 1, args.mode == "intra")],
        context=f"--mode {args.mode}",
    )

    seed_everything(args.seed)
    device = torch.device(f"cuda:{local_rank}")

    # Build logarithmically-spaced byte sizes (powers-of-2 friendly)
    import numpy as np
    byte_sizes = np.unique(
        np.round(
            np.logspace(
                np.log2(args.min_bytes),
                np.log2(args.max_bytes),
                num=args.steps,
                base=2,
            )
        ).astype(int)
    ).tolist()

    measurements = []
    for nbytes in byte_sizes:
        # Float32 elements
        num_elems = max(1, nbytes // 4)
        tensor = torch.zeros(num_elems, dtype=torch.float32, device=device)

        times = pingpong_timed(rank, tensor, args.warmup, args.trials)

        if rank == 0:
            measurements.append({
                "params": {
                    "message_bytes": num_elems * 4,
                    "num_elements": num_elems,
                    "mode": args.mode,
                },
                "trials_seconds": times,
            })
            print(
                f"[pingpong] {num_elems * 4:>10} bytes | "
                f"median {1e3 * float(sorted(times)[len(times)//2]):.3f} ms"
            )

        dist.barrier()

    if rank == 0:
        payload = {
            "benchmark": "pingpong",
            "metadata": collect_metadata(),
            "config": {
                "min_bytes": args.min_bytes,
                "max_bytes": args.max_bytes,
                "steps": args.steps,
                "warmup": args.warmup,
                "trials": args.trials,
                "mode": args.mode,
                "world_size": world_size,
                "seed": args.seed,
            },
            "measurements": measurements,
        }
        write_result(args.output, payload)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
