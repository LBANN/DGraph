"""Benchmark 1.2 — Intra/Inter Concurrency Check.

Verifies that T_both / max(T_intra, T_inter) ≈ 1, i.e. that NVLink and
InfiniBand transfers overlap when issued simultaneously.

Requires exactly 4 ranks across 2 nodes:
    Node A: rank 0 (A0), rank 1 (A1)
    Node B: rank 2 (B0), rank 3 (B1)

Three conditions at a fixed message size:
    1. intra-only  — A0↔A1 and B0↔B1  (no cross-node traffic)
    2. inter-only  — A0↔B0 and A1↔B1  (no intra-node traffic)
    3. concurrent  — all four exchanges in the same window (separate streams)

Each rank logs its own wall time per trial.  Rank 0 collects and writes JSON.

Usage::

    srun -N 2 --ntasks-per-node 2 python -m benchmarks.bench_concurrency \\
        --message-bytes 16777216 --warmup 20 --trials 100 \\
        --output data/concurrency.json --seed 42
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
# Helpers
# ---------------------------------------------------------------------------

def _exchange(tensor_send: torch.Tensor, tensor_recv: torch.Tensor,
              peer: int, stream: torch.cuda.Stream) -> None:
    """Non-blocking send+recv on *stream* with *peer*."""
    with torch.cuda.stream(stream):
        send_op = dist.P2POp(dist.isend, tensor_send, peer)
        recv_op = dist.P2POp(dist.irecv, tensor_recv, peer)
        reqs = dist.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()


def timed_window(rank: int,
                 send_buf: torch.Tensor,
                 recv_buf: torch.Tensor,
                 peer: int,
                 stream: torch.cuda.Stream,
                 warmup: int,
                 trials: int) -> list:
    """Time a single exchange window (one send+recv pair)."""
    for _ in range(warmup):
        _exchange(send_buf, recv_buf, peer, stream)
        torch.cuda.synchronize()
    dist.barrier()

    times = []
    for _ in range(trials):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _exchange(send_buf, recv_buf, peer, stream)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1_000.0)
    return times


def timed_concurrent(rank: int,
                     intra_send: torch.Tensor, intra_recv: torch.Tensor, intra_peer: int,
                     inter_send: torch.Tensor, inter_recv: torch.Tensor, inter_peer: int,
                     intra_stream: torch.cuda.Stream,
                     inter_stream: torch.cuda.Stream,
                     warmup: int,
                     trials: int) -> list:
    """Time intra and inter exchanges issued concurrently on separate streams."""
    for _ in range(warmup):
        _exchange(intra_send, intra_recv, intra_peer, intra_stream)
        _exchange(inter_send, inter_recv, inter_peer, inter_stream)
        torch.cuda.synchronize()
    dist.barrier()

    times = []
    for _ in range(trials):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _exchange(intra_send, intra_recv, intra_peer, intra_stream)
        _exchange(inter_send, inter_recv, inter_peer, inter_stream)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1_000.0)
    return times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Intra/inter concurrency benchmark")
    p.add_argument("--message-bytes", type=int, default=16_777_216)  # 16 MiB
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()

    if world_size != 4:
        raise ValueError(
            f"bench_concurrency requires exactly 4 ranks (got {world_size}).\n"
            "Layout: rank 0,1 on node A; rank 2,3 on node B."
        )

    # The peer maps below hardcode "ranks 0,1 on node A; ranks 2,3 on node B".
    # Nothing in the launch command enforces that, and running all 4 ranks on
    # one node would make the "inter" peers actually intra-node — silently
    # turning the overlap measurement (which the whole
    # T_comm = max(T_intra, T_inter) assumption rests on) into a comparison of
    # intra-node against intra-node.
    assert_placement(
        [(0, 1, True), (2, 3, True), (0, 2, False)],
        context="layout: ranks 0,1 on node A; ranks 2,3 on node B",
    )

    seed_everything(args.seed)
    device = torch.device(f"cuda:{local_rank}")

    num_elems = max(1, args.message_bytes // 4)
    send_buf = torch.randn(num_elems, dtype=torch.float32, device=device)
    recv_buf = torch.zeros(num_elems, dtype=torch.float32, device=device)

    # Intra-node peers: 0↔1, 2↔3
    # Inter-node peers: 0↔2, 1↔3
    intra_peer = {0: 1, 1: 0, 2: 3, 3: 2}[rank]
    inter_peer = {0: 2, 1: 3, 2: 0, 3: 1}[rank]

    intra_stream = torch.cuda.Stream(device=device)
    inter_stream = torch.cuda.Stream(device=device)

    intra_send = send_buf.clone()
    intra_recv = torch.zeros_like(recv_buf)
    inter_send = send_buf.clone()
    inter_recv = torch.zeros_like(recv_buf)

    # --- Condition 1: intra-only ---
    times_intra = timed_window(
        rank, intra_send, intra_recv, intra_peer, intra_stream,
        args.warmup, args.trials
    )
    dist.barrier()

    # --- Condition 2: inter-only ---
    times_inter = timed_window(
        rank, inter_send, inter_recv, inter_peer, inter_stream,
        args.warmup, args.trials
    )
    dist.barrier()

    # --- Condition 3: concurrent ---
    times_concurrent = timed_concurrent(
        rank,
        intra_send, intra_recv, intra_peer,
        inter_send, inter_recv, inter_peer,
        intra_stream, inter_stream,
        args.warmup, args.trials,
    )
    dist.barrier()

    # Gather per-rank times to rank 0
    def gather_times(times_local):
        obj = [None] * world_size
        dist.all_gather_object(obj, times_local)
        return obj

    intra_all = gather_times(times_intra)
    inter_all = gather_times(times_inter)
    conc_all = gather_times(times_concurrent)

    if rank == 0:
        measurements = [
            {
                "params": {"condition": "intra_only", "message_bytes": num_elems * 4},
                "per_rank_trials_seconds": intra_all,
            },
            {
                "params": {"condition": "inter_only", "message_bytes": num_elems * 4},
                "per_rank_trials_seconds": inter_all,
            },
            {
                "params": {"condition": "concurrent", "message_bytes": num_elems * 4},
                "per_rank_trials_seconds": conc_all,
            },
        ]
        payload = {
            "benchmark": "concurrency",
            "metadata": collect_metadata(),
            "config": {
                "message_bytes": args.message_bytes,
                "warmup": args.warmup,
                "trials": args.trials,
                "world_size": world_size,
                "seed": args.seed,
                "rank_layout": "rank 0,1 on node A; rank 2,3 on node B",
            },
            "measurements": measurements,
        }
        write_result(args.output, payload)
        print(
            f"[concurrency] intra median  = "
            f"{1e3*sorted(intra_all[0])[len(intra_all[0])//2]:.2f} ms | "
            f"inter median = "
            f"{1e3*sorted(inter_all[0])[len(inter_all[0])//2]:.2f} ms | "
            f"concurrent median = "
            f"{1e3*sorted(conc_all[0])[len(conc_all[0])//2]:.2f} ms"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
