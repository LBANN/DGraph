"""Benchmark 2.2 — Multi-GPU Crossover Point.

Sweeps graph size to identify the point at which distributing computation
across K GPUs overcomes the overhead of halo-exchange communication.

For each graph size N in the sweep:

  * **Single-GPU baseline**: rank 0 runs forward+backward on the *complete*
    graph on one GPU.  Measures raw compute cost T_comp(N).
  * **Multi-GPU distributed**: all K ranks execute partitioned
    forward+backward with halo exchange.  Measures
    T_comp(N/K) + T_comm.

The *crossover* N* is the smallest graph size where
T_single(N) > T_multi(K, N), i.e. where distributing first becomes
beneficial.

When ``--no-dist`` is set the script runs the single-GPU sweep only
(useful for characterising baseline compute without a multi-GPU
allocation).

Synthetic graphs:
    * ``erdos_renyi``  — Erdős-Rényi with ``--avg-degree`` expected degree
    * ``sbm``          — Stochastic Block Model; ``--sbm-inter-density``
                         controls the fraction of inter-block edges

Partitioners:
    * ``random``   — assign each vertex to a uniformly random rank
    * ``balanced`` — contiguous vertex blocks of equal size
    * ``metis``    — balanced k-way via pymetis (skipped if not installed)

Usage — distributed (torchrun)::

    torchrun --nnodes 1 --nproc_per_node 4 \\
        -m benchmarks.benchmark_crossover \\
        --graph erdos_renyi \\
        --graph-sizes 10000,100000,1000000,10000000 \\
        --avg-degree 20 --feature-dim 128 --model gcn \\
        --partitioner balanced --warmup 10 --trials 50 \\
        --output data/crossover_K8_F128_er_bal.json --seed 42

Usage — single-GPU baseline only::

    python -m benchmarks.benchmark_crossover --no-dist \\
        --graph erdos_renyi \\
        --graph-sizes 10000,100000,1000000,10000000 \\
        --avg-degree 20 --feature-dim 128 --model gcn \\
        --warmup 10 --trials 50 \\
        --output data/crossover_single_F128.json --seed 42
"""

import argparse

import numpy as np
import torch
import torch.distributed as dist
import gc

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

from DGraph.distributed import (
    HaloExchange,
    CommunicationPattern,
    build_communication_pattern,
)
from DGraph import Communicator


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Multi-GPU crossover point benchmark")
    p.add_argument("--graph", choices=["erdos_renyi", "sbm"], default="erdos_renyi")
    p.add_argument(
        "--graph-sizes",
        type=str,
        default="100,200,400,800,1000,2000,4000,8000,10000,16000,32000,100000,200000,400000",
        help="Comma-separated list of num_vertices values to sweep",
    )
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
    p.add_argument(
        "--no-dist",
        action="store_true",
        help="Run single-GPU sweep only (no distributed setup required)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gen_graph(graph_type, num_vertices, avg_degree, sbm_inter_density, seed):
    """Generate a synthetic graph reproducibly for a given graph size."""
    rng = np.random.default_rng(seed)
    if graph_type == "erdos_renyi":
        return gen_erdos_renyi(num_vertices, avg_degree, rng)
    else:
        return gen_sbm(num_vertices, avg_degree, sbm_inter_density, rng)


def _build_single_gpu_tensors(num_vertices, edges_np, F, model, device):
    """Allocate full-graph tensors on *device*.  May raise cuda.OutOfMemoryError.

    Returns (x, aux, layer, edge_attr) where ``aux`` is whatever the layer's
    second forward argument is: a ``[2, E]`` edge index for GCNLayer /
    EdgeConditionedLayer, or a prebuilt sparse ``[V, V]`` adjacency for
    GCNSpMMLayer (which does not take an edge index at all — see
    nn_layer_common.GCNSpMMLayer.forward). Built once here, outside the timed
    region, matching bench_compute.py's build_sparse_adj convention.
    """
    edge_t = torch.from_numpy(edges_np.T.copy()).long().to(device)
    x = torch.randn(num_vertices, F, device=device, requires_grad=True)
    if model == "gcn":
        layer = GCNLayer(F).to(device)
        edge_attr = None
        aux = edge_t
    elif model == "gcn_spmm":
        layer = GCNSpMMLayer(F).to(device)
        edge_attr = None
        aux = create_sparse_adj(edge_t, num_vertices, num_vertices, device)
    else:
        layer = EdgeConditionedLayer(F).to(device)
        edge_attr = torch.randn(edges_np.shape[0], F, device=device)
        aux = edge_t
    layer.train()
    return x, aux, layer, edge_attr


def _single_gpu_fn(x, aux, layer, edge_attr, model):
    """Return a zero-argument closure for single-GPU forward+backward."""
    if model == "gcn_spmm":

        def fn():
            out = layer(x, aux)
            out.sum().backward()
            if x.grad is not None:
                x.grad.zero_()

    elif edge_attr is None:

        def fn():
            out = layer(x, aux)
            out.sum().backward()
            if x.grad is not None:
                x.grad.zero_()

    else:

        def fn():
            out = layer(x, aux, edge_attr)
            out.sum().backward()
            if x.grad is not None:
                x.grad.zero_()

    return fn


def _multi_gpu_fn(x_local, comm_pattern, layer, halo_exchange, edge_attr, model):
    """Return a zero-argument closure for distributed forward+backward.

    ``comm_pattern.local_edge_list`` has shape ``[E, 2]``; we transpose it
    once to ``[2, E]`` as required by GCNLayer / EdgeConditionedLayer. For
    GCNSpMMLayer we instead build the rectangular
    ``[n_local, n_local + n_halo]`` sparse adjacency once here (outside the
    timed closure, same as the single-GPU path) since that layer takes the
    aggregation matrix directly rather than an edge index.
    """
    edge_index = comm_pattern.local_edge_list.T.contiguous()  # [2, E_local]

    if model == "gcn_spmm":
        n_local = comm_pattern.num_local_vertices
        n_halo = comm_pattern.num_halo_vertices
        adj = create_sparse_adj(
            edge_index, n_local, n_local + n_halo, x_local.device
        )

        def fn():
            recv_buf = halo_exchange(x_local, comm_pattern)
            x_aug = torch.cat([x_local, recv_buf], dim=0)
            out = layer(x_aug, adj)
            out.sum().backward()
            if x_local.grad is not None:
                x_local.grad.zero_()

    elif model == "gcn":

        def fn():
            recv_buf = halo_exchange(x_local, comm_pattern)
            x_aug = torch.cat([x_local, recv_buf], dim=0)

            out = layer(x_aug, edge_index)
            out.sum().backward()
            if x_local.grad is not None:
                x_local.grad.zero_()

    else:

        def fn():
            recv_buf = halo_exchange(x_local, comm_pattern)
            x_aug = torch.cat([x_local, recv_buf], dim=0)
            out = layer(x_aug, edge_index, edge_attr)
            out.sum().backward()
            if x_local.grad is not None:
                x_local.grad.zero_()

    return fn


# ---------------------------------------------------------------------------
# Single-GPU-only path
# ---------------------------------------------------------------------------


def run_no_dist(args, graph_sizes, F):
    """Sweep graph sizes on a single GPU and return the measurement list."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    measurements = []

    for num_vertices in graph_sizes:
        # Unique seed per graph size so each topology is reproducible independently
        graph_seed = args.seed + num_vertices
        edges_np = _gen_graph(
            args.graph,
            num_vertices,
            args.avg_degree,
            args.sbm_inter_density,
            graph_seed,
        )
        num_edges = int(edges_np.shape[0])

        times_single = None
        oom = False
        try:
            x, aux, layer, edge_attr = _build_single_gpu_tensors(
                num_vertices, edges_np, F, args.model, device
            )
            fn = _single_gpu_fn(x, aux, layer, edge_attr, args.model)
            times_single = cuda_timed(fn, warmup=args.warmup, trials=args.trials)
            # Free before next iteration
            del x, aux, layer
            if edge_attr is not None:
                del edge_attr
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            oom = True
            print(
                f"[crossover] OOM: N={num_vertices:,} exceeds single-GPU memory; "
                "skipping and continuing"
            )
            torch.cuda.empty_cache()

        med_str = (
            f"{1e3 * sorted(times_single)[len(times_single) // 2]:.2f} ms"
            if times_single
            else "OOM"
        )
        print(f"[crossover] no-dist N={num_vertices:>12,}  single={med_str}")

        measurements.append(
            {
                "params": {
                    "num_vertices": num_vertices,
                    "num_edges": num_edges,
                    "avg_degree": args.avg_degree,
                    "feature_dim": F,
                    "model": args.model,
                    "graph": args.graph,
                },
                "single_gpu_trials_seconds": times_single,
                "single_gpu_oom": oom,
                "multi_gpu_trials_seconds_rank0": None,
                "multi_gpu_trials_seconds_max": None,
                "per_rank_stats": None,
                "world_size": 1,
            }
        )

    return measurements


# ---------------------------------------------------------------------------
# Distributed path
# ---------------------------------------------------------------------------


def run_distributed(args, graph_sizes, F, rank, world_size, local_rank):
    """Run one crossover measurement per graph size using all K ranks."""
    device = torch.device(f"cuda:{local_rank}")

    comm = Communicator(backend="nccl")

    ranks_per_node = get_ranks_per_node()

    measurements = []

    for num_vertices in graph_sizes:
        halo_exchange = HaloExchange(comm=comm)
        # All ranks generate *identical* graph topology (same seed per size)
        graph_seed = args.seed + num_vertices
        edges_np = _gen_graph(
            args.graph,
            num_vertices,
            args.avg_degree,
            args.sbm_inter_density,
            graph_seed,
        )

        # All ranks compute *identical* partition assignment
        rng_part = np.random.default_rng(args.seed + 1 + num_vertices)
        if args.partitioner == "random":
            assignment_np = partition_random(num_vertices, world_size, rng_part)
        elif args.partitioner == "balanced":
            assignment_np = partition_balanced(num_vertices, world_size)
        else:
            assignment_np = partition_metis(num_vertices, world_size, edges_np)

        # Move to GPU for DGraph's build_communication_pattern (collective)
        edges_t = torch.from_numpy(edges_np).long().to(device)  # [E, 2]
        partitioning_t = torch.from_numpy(assignment_np).long().to(device)  # [V]

        # Collective: all ranks call this in sync (internally calls dist.all_gather)
        comm_pattern = build_communication_pattern(
            edges_t, partitioning_t, rank, world_size
        )

        n_local = comm_pattern.num_local_vertices
        n_halo = comm_pattern.num_halo_vertices
        n_local_edges = comm_pattern.local_edge_list.shape[0]

        x_local = torch.randn(n_local, F, device=device, requires_grad=True)
        edge_attr_dist = (
            torch.randn(n_local_edges, F, device=device)
            if args.model == "edge"
            else None
        )
        if args.model == "gcn":
            layer_dist = GCNLayer(F).to(device)
        elif args.model == "gcn_spmm":
            layer_dist = GCNSpMMLayer(F).to(device)
        else:
            layer_dist = EdgeConditionedLayer(F).to(device)
        layer_dist.train()

        fn_multi = _multi_gpu_fn(
            x_local, comm_pattern, layer_dist, halo_exchange, edge_attr_dist, args.model
        )

        # ---- Time multi-GPU (all ranks participate) ----
        dist.barrier()
        times_multi_local = cuda_timed(fn_multi, warmup=args.warmup, trials=args.trials)
        dist.barrier()

        # Gather timings and partition stats from all ranks to rank 0
        all_times_multi = [None] * world_size
        dist.all_gather_object(all_times_multi, times_multi_local)

        intra_halo, inter_halo = intra_inter_halo(comm_pattern, ranks_per_node)
        stats_local = {
            "rank": rank,
            "n_local": n_local,
            "n_halo": n_halo,
            "n_local_edges": n_local_edges,
            "intra_halo_size": intra_halo,
            "inter_halo_size": inter_halo,
            "c_intra_bytes": intra_halo * F * 4,
            "c_inter_bytes": inter_halo * F * 4,
            "send_total": int(comm_pattern.send_offset[-1].item()),
            "recv_total": int(comm_pattern.recv_offset[-1].item()),
            "trials_seconds": times_multi_local,
        }
        all_stats = [None] * world_size
        dist.all_gather_object(all_stats, stats_local)

        # Free GPU memory before the single-GPU run
        del (
            x_local,
            edges_t,
            partitioning_t,
            layer_dist,
            halo_exchange,
            comm_pattern,
            fn_multi,
        )
        if edge_attr_dist is not None:
            del edge_attr_dist
        gc.collect()
        torch.cuda.empty_cache()

        # ---- Rank 0: time single-GPU baseline (non-collective) ----
        times_single = None
        single_oom = False
        if rank == 0:
            gc.collect()
            torch.cuda.empty_cache()
            edges_s = _gen_graph(
                args.graph,
                num_vertices,
                args.avg_degree,
                args.sbm_inter_density,
                graph_seed,
            )
            try:
                x_s, aux_s, layer_s, edge_attr_s = _build_single_gpu_tensors(
                    num_vertices, edges_s, F, args.model, device
                )
                fn_single = _single_gpu_fn(x_s, aux_s, layer_s, edge_attr_s, args.model)
                times_single = cuda_timed(
                    fn_single, warmup=args.warmup, trials=args.trials
                )
                del x_s, aux_s, layer_s, fn_single
                if edge_attr_s is not None:
                    del edge_attr_s
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                single_oom = True
                print(
                    f"[crossover] rank 0 OOM: N={num_vertices:,} exceeds single-GPU memory"
                )
                torch.cuda.empty_cache()

        # Sync all ranks after rank 0 finishes its single-GPU benchmark
        dist.barrier()

        if rank == 0:
            n_trials = len(times_multi_local)
            # Wall time = max latency across all ranks per trial
            times_multi_max = [
                max(all_times_multi[r][i] for r in range(world_size))
                for i in range(n_trials)
            ]

            med_multi = sorted(times_multi_max)[n_trials // 2]
            if times_single:
                med_single = sorted(times_single)[len(times_single) // 2]
                speedup = med_single / med_multi if med_multi > 0 else float("nan")
            else:
                med_single = float("nan")
                speedup = float("nan")

            print(
                f"[crossover] K={world_size:>3}  N={num_vertices:>12,}  "
                f"single={1e3*med_single:.2f}ms  "
                f"multi={1e3*med_multi:.2f}ms  "
                f"speedup={speedup:.2f}x"
            )

            measurements.append(
                {
                    "params": {
                        "num_vertices": num_vertices,
                        "num_global_edges": int(edges_np.shape[0]),
                        "avg_degree": args.avg_degree,
                        "feature_dim": F,
                        "model": args.model,
                        "graph": args.graph,
                        "partitioner": args.partitioner,
                        "world_size": world_size,
                        "ranks_per_node": ranks_per_node,
                    },
                    "single_gpu_trials_seconds": times_single,
                    "single_gpu_oom": single_oom,
                    "multi_gpu_trials_seconds_rank0": times_multi_local,
                    "multi_gpu_trials_seconds_max": times_multi_max,
                    "per_rank_stats": all_stats,
                }
            )

    return measurements, ranks_per_node


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    graph_sizes = [int(s) for s in args.graph_sizes.split(",")]
    F = args.feature_dim

    if args.no_dist:
        measurements = run_no_dist(args, graph_sizes, F)
        payload = {
            "benchmark": "crossover",
            "metadata": collect_metadata(),
            "config": {
                "graph": args.graph,
                "graph_sizes": graph_sizes,
                "avg_degree": args.avg_degree,
                "sbm_inter_density": args.sbm_inter_density,
                "feature_dim": F,
                "model": args.model,
                "partitioner": "none",
                "world_size": 1,
                "ranks_per_node": 1,
                "warmup": args.warmup,
                "trials": args.trials,
                "seed": args.seed,
                "mode": "single_gpu_only",
            },
            "measurements": measurements,
        }
        write_result(args.output, payload)
        return

    # ---- Distributed run ----
    rank, world_size, local_rank = setup_distributed()
    seed_everything(args.seed + rank)

    measurements, ranks_per_node = run_distributed(
        args, graph_sizes, F, rank, world_size, local_rank
    )

    if rank == 0:
        payload = {
            "benchmark": "crossover",
            "metadata": collect_metadata(),
            "config": {
                "graph": args.graph,
                "graph_sizes": graph_sizes,
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
                "mode": "distributed",
            },
            "measurements": measurements,
        }
        write_result(args.output, payload)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
