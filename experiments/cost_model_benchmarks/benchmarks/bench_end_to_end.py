"""Benchmark 2.1 — End-to-End Halo Exchange.

Measures full GNN layer wall time (forward + backward) across a sweep of
configurations on the full multi-node setup.  Intended to be run as a SLURM
array job with one invocation per (K, F, graph) combination.

This module contains a self-contained minimal halo-exchange implementation
(no dependency on the DGraph production library) so the benchmark remains
isolated and portable.

Synthetic graphs:
    * ``erdos_renyi``  — Erdős-Rényi with ``--avg-degree`` expected degree
    * ``sbm``          — Stochastic Block Model; ``--sbm-inter-density`` controls
                         the fraction of inter-block edges (topology ablation)

Partitioners:
    * ``random``   — assign each vertex to a uniformly random rank
    * ``balanced`` — contiguous vertex blocks of equal size
    * ``metis``    — balanced k-way via pymetis (skipped if not installed)

The benchmark logs, for every run:
    * world_size K, feature dim F, graph type, partitioner
    * per-rank partition statistics:
        intra_halo_size  — halo vertices on the same node
        inter_halo_size  — halo vertices on different nodes
        c_intra, c_inter — communication volumes (bytes)
    * per-trial layer times from rank 0 (and per-rank times for completeness)

Usage::

    torchrun --nnodes 2 --nproc_per_node 4 \\
        -m benchmarks.bench_end_to_end \\
        --graph erdos_renyi --num-vertices 100000 --avg-degree 20 \\
        --feature-dim 128 --model gcn --partitioner balanced \\
        --warmup 10 --trials 50 \\
        --output data/e2e_K8_F128_er_bal.json --seed 42
"""

import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from benchmarks.common import (
    collect_metadata,
    cuda_timed,
    seed_everything,
    setup_distributed,
    write_result,
)
from benchmarks.graph_data_common import (
    gen_erdos_renyi,
    gen_sbm,
    partition_balanced,
    partition_metis,
    partition_random,
)
from benchmarks.nn_layer_common import GCNLayer, EdgeConditionedLayer


class MinimalHaloExchange(torch.autograd.Function):
    """Forward: gather boundary features → all_to_all → populate recv buffer.
    Backward: reverse the transfer to accumulate gradients.
    """

    @staticmethod
    def forward(ctx, x_local, send_idx_flat, send_counts, recv_counts, world_size):
        # Gather send buffer
        send_buf = x_local[send_idx_flat]  # [total_send, F]

        # Split by destination rank
        send_list = list(send_buf.split(send_counts, dim=0))
        recv_list = [
            torch.zeros(
                rc, x_local.shape[1], dtype=x_local.dtype, device=x_local.device
            )
            for rc in recv_counts
        ]

        dist.all_to_all(recv_list, send_list)

        recv_buf = (
            torch.cat(recv_list, dim=0)
            if sum(recv_counts) > 0
            else torch.zeros(0, x_local.shape[1], device=x_local.device)
        )

        ctx.save_for_backward(send_idx_flat)
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.world_size = world_size
        ctx.n_local = x_local.shape[0]
        ctx.feature_dim = x_local.shape[1]
        ctx.device = x_local.device

        return recv_buf

    @staticmethod
    def backward(ctx, grad_recv):
        (send_idx_flat,) = ctx.saved_tensors
        send_counts = ctx.send_counts
        recv_counts = ctx.recv_counts
        world_size = ctx.world_size
        n_local = ctx.n_local
        F = ctx.feature_dim
        device = ctx.device

        # Reverse: recv_counts become send_counts and vice versa
        grad_recv_list = (
            list(grad_recv.split(recv_counts, dim=0))
            if grad_recv.shape[0] > 0
            else [torch.zeros(0, F, device=device)] * world_size
        )
        grad_send_list = [torch.zeros(sc, F, device=device) for sc in send_counts]

        dist.all_to_all(grad_send_list, grad_recv_list)

        grad_send = torch.cat(grad_send_list, dim=0)

        # Scatter-add back to local vertices
        grad_x_local = torch.zeros(n_local, F, device=device, dtype=grad_recv.dtype)
        grad_x_local.scatter_add_(
            0,
            send_idx_flat.unsqueeze(1).expand_as(grad_send),
            grad_send,
        )
        return grad_x_local, None, None, None, None


# ===========================================================================
# Main
# ===========================================================================


def parse_args():
    p = argparse.ArgumentParser(description="End-to-end halo exchange benchmark")
    p.add_argument("--graph", choices=["erdos_renyi", "sbm"], default="erdos_renyi")
    p.add_argument("--num-vertices", type=int, default=100_000)
    p.add_argument("--avg-degree", type=float, default=20.0)
    p.add_argument(
        "--sbm-inter-density",
        type=float,
        default=0.1,
        help="Fraction of inter-block edges for SBM graphs",
    )
    p.add_argument("--feature-dim", type=int, default=128)
    p.add_argument("--model", choices=["gcn", "edge"], default="gcn")
    p.add_argument(
        "--partitioner", choices=["random", "balanced", "metis"], default="balanced"
    )
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    seed_everything(args.seed + rank)  # per-rank seed for graph generation
    rng = np.random.default_rng(args.seed)  # shared seed for graph topology
    device = torch.device(f"cuda:{local_rank}")
    F = args.feature_dim

    # --- Generate graph on all ranks (same seed → identical graph) ---
    if args.graph == "erdos_renyi":
        edges = gen_erdos_renyi(args.num_vertices, args.avg_degree, rng)
    else:
        edges = gen_sbm(args.num_vertices, args.avg_degree, args.sbm_inter_density, rng)

    # --- Partition ---
    rng_part = np.random.default_rng(args.seed + 1)
    if args.partitioner == "random":
        assignment = partition_random(args.num_vertices, world_size, rng_part)
    elif args.partitioner == "balanced":
        assignment = partition_balanced(args.num_vertices, world_size)
    else:
        assignment = partition_metis(args.num_vertices, world_size, edges)

    # --- Build local comm pattern ---
    pattern = build_local_comm_pattern(edges, assignment, rank, world_size)
    n_local = pattern["n_local"]
    n_halo = pattern["n_halo"]
    edge_index = pattern["edge_index"].to(device)

    send_counts = pattern["send_counts"]
    recv_counts = pattern["recv_counts"]
    send_idx_flat = (
        torch.cat(
            [torch.tensor(s, dtype=torch.long) for s in pattern["send_idx_by_rank"]]
        ).to(device)
        if sum(send_counts) > 0
        else torch.zeros(0, dtype=torch.long, device=device)
    )

    # --- Model ---
    if args.model == "gcn":
        layer = GCNLayer(F).to(device)
    else:
        layer = EdgeConditionedLayer(F).to(device)
    layer.train()

    # --- Synthetic local node features ---
    x_local = torch.randn(n_local, F, device=device, requires_grad=True)
    edge_attr = (
        torch.randn(edge_index.shape[1], F, device=device)
        if args.model == "edge"
        else None
    )

    # --- Timed forward + backward ---
    def one_layer():
        # Forward halo exchange
        recv_buf = MinimalHaloExchange.apply(
            x_local, send_idx_flat, send_counts, recv_counts, world_size
        )
        # Augment: local + halo
        x_aug = torch.cat([x_local, recv_buf], dim=0)
        # Message passing
        if args.model == "gcn":
            out = layer(x_aug, edge_index)
        else:
            out = layer(x_aug, edge_index, edge_attr)
        # Backward
        loss = out.sum()
        loss.backward()
        if x_local.grad is not None:
            x_local.grad.zero_()

    # Barrier before timing
    dist.barrier()
    times_local = cuda_timed(one_layer, warmup=args.warmup, trials=args.trials)
    dist.barrier()

    # Gather per-rank times and stats to rank 0
    stats_local = {
        "rank": rank,
        "n_local": n_local,
        "n_halo": n_halo,
        "intra_halo_size": pattern["intra_halo_size"],
        "inter_halo_size": pattern["inter_halo_size"],
        "c_intra_bytes": pattern["intra_halo_size"] * F * 4,
        "c_inter_bytes": pattern["inter_halo_size"] * F * 4,
        "send_total": sum(send_counts),
        "recv_total": sum(recv_counts),
        "trials_seconds": times_local,
    }

    all_stats = [None] * world_size
    dist.all_gather_object(all_stats, stats_local)

    if rank == 0:
        med = sorted(times_local)[len(times_local) // 2]
        print(
            f"[e2e] K={world_size} F={F} {args.graph}/{args.partitioner}/{args.model} "
            f"n_local={n_local} n_halo={n_halo} "
            f"median {1e3*med:.2f} ms"
        )
        payload = {
            "benchmark": "end_to_end",
            "metadata": collect_metadata(),
            "config": {
                "graph": args.graph,
                "num_vertices": args.num_vertices,
                "avg_degree": args.avg_degree,
                "sbm_inter_density": args.sbm_inter_density,
                "feature_dim": F,
                "model": args.model,
                "partitioner": args.partitioner,
                "world_size": world_size,
                "ranks_per_node": pattern["ranks_per_node"],
                "warmup": args.warmup,
                "trials": args.trials,
                "seed": args.seed,
            },
            "measurements": [
                {
                    "params": {
                        "world_size": world_size,
                        "feature_dim": F,
                        "graph": args.graph,
                        "partitioner": args.partitioner,
                        "model": args.model,
                    },
                    "rank0_trials_seconds": times_local,
                    "per_rank_stats": all_stats,
                }
            ],
        }
        write_result(args.output, payload)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
