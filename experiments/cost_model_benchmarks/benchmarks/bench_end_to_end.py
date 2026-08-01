"""Benchmark 2.1 — End-to-End Halo Exchange.

Measures full GNN layer wall time (forward + backward) across a sweep of
configurations on the full multi-node setup.  Intended to be run as a SLURM
array job with one invocation per (K, F, graph) combination.

Uses DGraph's production ``build_communication_pattern``/``HaloExchange``
(``DGraph.distributed``) for graph partitioning/halo-exchange, matching
bench_crossover.py, so the benchmark measures the real communication path.

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
    get_ranks_per_node,
    intra_inter_halo,
)
from benchmarks.nn_layer_common import (
    GCNLayer,
    EdgeConditionedLayer,
    GCNSpMMLayer,
    create_sparse_adj,
)

from DGraph.distributed import HaloExchange, CommunicationPattern, build_communication_pattern
from DGraph import Communicator

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
    p.add_argument("--model", choices=["gcn", "edge", "gcn_spmm"], default="gcn")
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

    # --- Build local comm pattern (collective: internally calls dist.all_gather) ---
    edges_t = torch.from_numpy(edges).long().to(device)  # [E, 2]
    assignment_t = torch.from_numpy(assignment).long().to(device)  # [V]
    comm_pattern = build_communication_pattern(edges_t, assignment_t, rank, world_size)

    n_local = comm_pattern.num_local_vertices
    n_halo = comm_pattern.num_halo_vertices
    edge_index = comm_pattern.local_edge_list.T.contiguous()  # [2, E_local]

    ranks_per_node = get_ranks_per_node()

    # --- Model ---
    if args.model == "gcn":
        layer = GCNLayer(F).to(device)
    elif args.model == "gcn_spmm":
        layer = GCNSpMMLayer(F).to(device)
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
    # GCNSpMMLayer takes the prebuilt rectangular [n_local, n_local+n_halo]
    # aggregation matrix instead of an edge index (see nn_layer_common) —
    # built once here, outside the timed closure, matching bench_crossover.py.
    adj = (
        create_sparse_adj(edge_index, n_local, n_local + n_halo, device)
        if args.model == "gcn_spmm"
        else None
    )

    comm = Communicator(backend="nccl")
    halo_exchange = HaloExchange(comm=comm)

    # --- Timed forward + backward ---
    def one_layer():
        # Forward halo exchange
        recv_buf = halo_exchange(x_local, comm_pattern)
        # Augment: local + halo
        x_aug = torch.cat([x_local, recv_buf], dim=0)
        # Message passing
        if args.model == "gcn":
            out = layer(x_aug, edge_index)
        elif args.model == "gcn_spmm":
            out = layer(x_aug, adj)
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
    intra_halo, inter_halo = intra_inter_halo(comm_pattern, ranks_per_node)
    stats_local = {
        "rank": rank,
        "n_local": n_local,
        "n_halo": n_halo,
        "intra_halo_size": intra_halo,
        "inter_halo_size": inter_halo,
        "c_intra_bytes": intra_halo * F * 4,
        "c_inter_bytes": inter_halo * F * 4,
        "n_edges_local": edge_index.shape[1],
        "send_total": int(comm_pattern.send_offset[-1].item()),
        "recv_total": int(comm_pattern.recv_offset[-1].item()),
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
                "ranks_per_node": ranks_per_node,
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
