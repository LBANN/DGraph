"""Benchmark 1.2 — Intra/Inter Concurrency Check.

Measures T_both / max(T_intra, T_inter), i.e. whether NVLink and InfiniBand
transfers overlap when issued together. A ratio near 1.0 justifies
``T_comm = max(T_intra, T_inter)`` in the cost model; a ratio near
``(T_intra + T_inter) / max(...)`` means they serialize and the tiers must be
added instead.

Requires exactly 4 ranks across 2 nodes:
    Node A: rank 0 (A0), rank 1 (A1)
    Node B: rank 2 (B0), rank 3 (B1)

Four conditions at a fixed message size:
    1. intra-only        — A0↔A1 and B0↔B1  (no cross-node traffic)
    2. inter-only        — A0↔B0 and A1↔B1  (no intra-node traffic)
    3. concurrent        — both, posted as ONE batch_isend_irecv group call,
                           matching how DGraph's halo exchange issues a single
                           all_to_all_single over every peer
    4. sequential_calls  — both, posted as two group calls with a wait in
                           between; the anti-pattern, measured so the cost of
                           not batching is quantified rather than assumed

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

# NOTE ON WHAT CHANGED AND WHY
#
# The original version of this benchmark issued the intra and inter exchanges
# as two separate ``batch_isend_irecv`` group calls, each followed immediately
# by ``req.wait()``, with the two calls placed on different CUDA streams. It
# reported T_concurrent / max(T_intra, T_inter) = 1.60 -- i.e. no overlap --
# and the cost model was briefly changed to add the two tiers because of it.
#
# That measurement was an artefact of how the ops were posted, not a property
# of the hardware:
#
#   1. ``req.wait()`` sat inside the per-peer helper, so the intra exchange was
#      joined onto the calling stream *before* the inter exchange was even
#      enqueued. The host serialized the two before NCCL saw them.
#   2. Both went through the default process group. NCCL orders operations on a
#      communicator, so two group calls become two kernel launches executed in
#      issue order. Putting them on different CUDA streams does not help: the
#      user-facing stream is not where NCCL performs the transfer.
#   3. Most importantly, it did not match the code being modelled.
#      DGraph's halo exchange issues a single ``dist.all_to_all_single`` over
#      every peer at once (NCCLBackendEngine.py), which NCCL schedules across
#      channels and *can* overlap NVLink and IB portions of.
#
# The end-to-end data agrees with (3): at K=8 the model's over-prediction is
# exactly T_intra (ratio 0.98-1.01 at the two larger sizes), i.e. intra-node
# traffic is fully hidden under the inter-node transfer in the real exchange.
#
# So the "concurrent" condition below now posts **one** group call containing
# both peers' ops -- matching the real exchange. The old two-call pattern is
# kept as a separate "sequential_calls" condition, because the gap between the
# two is a directly useful number: it is what NCCL leaves on the table when
# transfers are not batched into a single collective, and therefore part of
# the motivation for a finer-grained backend (NVSHMEM) that can issue
# NVLink stores and IB puts from within one kernel.
#
# CUDA streams are gone entirely: they gave the appearance of controlling
# concurrency while having no effect on it.

def _ops_for(peer: int, send_buf: torch.Tensor, recv_buf: torch.Tensor) -> list:
    """Build the (isend, irecv) P2POp pair for one peer, unposted."""
    return [
        dist.P2POp(dist.isend, send_buf, peer),
        dist.P2POp(dist.irecv, recv_buf, peer),
    ]


def _post_one_group(ops: list) -> None:
    """Post *all* ops as a single NCCL group call and wait for completion.

    One group call is the point: NCCL is free to run the enclosed transfers
    concurrently across channels. Splitting them across several calls forbids
    that, regardless of streams.
    """
    for req in dist.batch_isend_irecv(ops):
        req.wait()


def _post_separate_groups(op_groups: list) -> None:
    """Post each op list as its own group call, waiting between them.

    Deliberately the anti-pattern, retained for measurement.
    """
    for ops in op_groups:
        for req in dist.batch_isend_irecv(ops):
            req.wait()


def timed(post_fn, warmup: int, trials: int) -> list:
    """Time a comm window built by *post_fn* (a zero-arg closure)."""
    for _ in range(warmup):
        post_fn()
        torch.cuda.synchronize()
    dist.barrier()

    times = []
    for _ in range(trials):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        post_fn()
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

    intra_send = send_buf.clone()
    intra_recv = torch.zeros_like(recv_buf)
    inter_send = send_buf.clone()
    inter_recv = torch.zeros_like(recv_buf)

    intra_ops = lambda: _ops_for(intra_peer, intra_send, intra_recv)
    inter_ops = lambda: _ops_for(inter_peer, inter_send, inter_recv)

    # --- Condition 1: intra-only ---
    times_intra = timed(
        lambda: _post_one_group(intra_ops()), args.warmup, args.trials
    )
    dist.barrier()

    # --- Condition 2: inter-only ---
    times_inter = timed(
        lambda: _post_one_group(inter_ops()), args.warmup, args.trials
    )
    dist.barrier()

    # --- Condition 3: concurrent, ONE group call (matches the real exchange) ---
    times_concurrent = timed(
        lambda: _post_one_group(intra_ops() + inter_ops()),
        args.warmup, args.trials,
    )
    dist.barrier()

    # --- Condition 4: the anti-pattern, two group calls with a wait between ---
    times_sequential = timed(
        lambda: _post_separate_groups([intra_ops(), inter_ops()]),
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
    seq_all = gather_times(times_sequential)

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
                "params": {"condition": "concurrent", "message_bytes": num_elems * 4,
                           "posting": "single batch_isend_irecv group call"},
                "per_rank_trials_seconds": conc_all,
            },
            {
                "params": {"condition": "sequential_calls", "message_bytes": num_elems * 4,
                           "posting": "two group calls, wait between"},
                "per_rank_trials_seconds": seq_all,
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
        def med(times_all):
            t = sorted(times_all[0])
            return t[len(t) // 2]

        t_intra, t_inter = med(intra_all), med(inter_all)
        t_conc, t_seq = med(conc_all), med(seq_all)
        print(
            f"[concurrency] intra = {1e3*t_intra:.3f} ms | "
            f"inter = {1e3*t_inter:.3f} ms | "
            f"concurrent (1 group call) = {1e3*t_conc:.3f} ms | "
            f"sequential (2 calls) = {1e3*t_seq:.3f} ms"
        )
        # The overlap verdict the cost model depends on: ratio ~1.0 means the
        # tiers overlap (T_comm = max(...)); ratio ~ (T_intra+T_inter)/max
        # means they serialize (T_comm = T_intra + T_inter).
        print(
            f"[concurrency] T_concurrent / max(T_intra, T_inter) = "
            f"{t_conc / max(t_intra, t_inter):.2f}   "
            f"(1.00 = perfect overlap, "
            f"{(t_intra + t_inter) / max(t_intra, t_inter):.2f} = full serialization)"
        )
        print(
            f"[concurrency] cost of NOT batching into one group call: "
            f"{1e3*(t_seq - t_conc):+.3f} ms ({100*(t_seq/t_conc - 1):+.1f}%)"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
