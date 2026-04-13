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


# ===========================================================================
# Synthetic graph generators
# ===========================================================================

def gen_erdos_renyi(num_vertices: int, avg_degree: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Return edge array of shape [E, 2] (src, dst) for an Erdős-Rényi digraph."""
    num_edges = int(num_vertices * avg_degree)
    src = rng.integers(0, num_vertices, size=num_edges)
    dst = rng.integers(0, num_vertices, size=num_edges)
    return np.stack([src, dst], axis=1)


def gen_sbm(num_vertices: int, avg_degree: float, inter_density: float,
            rng: np.random.Generator) -> np.ndarray:
    """Return edges for a Stochastic Block Model graph.

    Vertices are split into blocks of equal size (one per rank for convenience,
    though the actual partitioning is a separate step).  The ratio of
    intra-block to inter-block edges is controlled by *inter_density*.
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 4
    block_size = num_vertices // world_size
    edges = []
    target_edges = int(num_vertices * avg_degree)

    intra_edges = int(target_edges * (1.0 - inter_density))
    inter_edges = target_edges - intra_edges

    # Intra-block edges
    for b in range(world_size):
        start = b * block_size
        end = start + block_size
        n = intra_edges // world_size
        s = rng.integers(start, end, size=n)
        d = rng.integers(start, end, size=n)
        edges.append(np.stack([s, d], axis=1))

    # Inter-block edges
    s = rng.integers(0, num_vertices, size=inter_edges)
    d = rng.integers(0, num_vertices, size=inter_edges)
    # Force cross-block by offsetting dst block
    d_block = (s // block_size + 1 + rng.integers(0, world_size - 1,
                                                    size=inter_edges)) % world_size
    d = d_block * block_size + rng.integers(0, block_size, size=inter_edges)
    d = np.clip(d, 0, num_vertices - 1)
    edges.append(np.stack([s, d], axis=1))

    return np.concatenate(edges, axis=0)


# ===========================================================================
# Partitioners
# ===========================================================================

def partition_random(num_vertices: int, world_size: int,
                     rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, world_size, size=num_vertices).astype(np.int64)


def partition_balanced(num_vertices: int, world_size: int) -> np.ndarray:
    return np.floor(np.arange(num_vertices) * world_size / num_vertices).astype(np.int64)


def partition_metis(num_vertices: int, world_size: int,
                    edges: np.ndarray) -> np.ndarray:
    try:
        import pymetis
    except ImportError:
        raise RuntimeError(
            "pymetis is not installed. Install it with: pip install pymetis\n"
            "Or use --partitioner random or --partitioner balanced."
        )
    # Build adjacency list for pymetis
    adj = [[] for _ in range(num_vertices)]
    for s, d in edges:
        adj[s].append(int(d))
        adj[d].append(int(s))
    _, membership = pymetis.part_graph(world_size, adjacency=adj)
    return np.array(membership, dtype=np.int64)


# ===========================================================================
# Minimal halo-exchange infrastructure
# ===========================================================================

def build_local_comm_pattern(edges: np.ndarray, assignment: np.ndarray,
                              rank: int, world_size: int):
    """Compute the local communication pattern for this rank.

    Returns a dict with:
        local_vertices      — np.ndarray of vertex IDs owned by this rank
        local_edge_index    — torch.Tensor [2, E_local] with local vertex IDs
                              remapped so that 0..n_local-1 are owned vertices
                              and n_local..n_local+n_halo-1 are halo vertices
        send_counts         — list[int] of length world_size: vertices to send
        recv_counts         — list[int] of length world_size: vertices to recv
        send_idx            — local indices (into local_vertices) to send per rank
        halo_global_ids     — global vertex IDs of halo vertices, in recv order
        intra_halo_size     — halo vertices from same node (ranks sharing node)
        inter_halo_size     — halo vertices from remote nodes
        ranks_per_node      — int (derived from LOCAL_RANK / RANK relationship)
    """
    local_mask = assignment == rank
    local_vertices = np.where(local_mask)[0]
    n_local = len(local_vertices)

    # Global -> local index map
    g2l = {int(v): i for i, v in enumerate(local_vertices)}

    # Find edges where dst is local
    local_dst_mask = np.isin(edges[:, 1], local_vertices)
    local_edges = edges[local_dst_mask]

    # Halo: src vertices not owned by this rank
    halo_src_mask = ~np.isin(local_edges[:, 0], local_vertices)
    halo_global = np.unique(local_edges[halo_src_mask, 0])

    # Group halo vertices by owning rank
    halo_owners = assignment[halo_global]
    recv_by_rank = []
    halo_order = []
    for r in range(world_size):
        verts = halo_global[halo_owners == r]
        recv_by_rank.append(verts)
        halo_order.extend(verts.tolist())
    halo_order = np.array(halo_order, dtype=np.int64)

    # Global halo id -> local halo index
    halo_g2l = {int(v): n_local + i for i, v in enumerate(halo_order)}
    all_g2l = {**g2l, **halo_g2l}

    # Find which local vertices other ranks need (send pattern)
    # We exchange recv_counts via all_to_all to learn send_counts
    recv_counts = [len(rv) for rv in recv_by_rank]

    # Build send: for each rank r, which of our local vertices does r need?
    # We do a global exchange of halo_global per rank
    all_recv = [None] * world_size
    dist.all_gather_object(all_recv, halo_order.tolist())

    send_idx_by_rank = []
    for r in range(world_size):
        needed = np.array(all_recv[r], dtype=np.int64)
        owned_mask = assignment[needed] == rank if len(needed) > 0 else np.array([], dtype=bool)
        owned = needed[owned_mask] if len(needed) > 0 else np.array([], dtype=np.int64)
        # Map to local indices
        local_idxs = np.array([g2l[int(v)] for v in owned], dtype=np.int64)
        send_idx_by_rank.append(local_idxs)

    send_counts = [len(s) for s in send_idx_by_rank]

    # Remap edges to local indices
    valid_edge_mask = np.array([
        (int(s) in all_g2l) and (int(d) in all_g2l)
        for s, d in local_edges
    ])
    local_edges_valid = local_edges[valid_edge_mask]
    if len(local_edges_valid) > 0:
        remapped_src = np.array([all_g2l[int(s)] for s in local_edges_valid[:, 0]])
        remapped_dst = np.array([all_g2l[int(d)] for d in local_edges_valid[:, 1]])
        edge_index = torch.tensor(
            np.stack([remapped_src, remapped_dst], axis=0), dtype=torch.long
        )
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Compute intra / inter halo sizes
    ranks_per_node = int(os.environ.get("LOCAL_WORLD_SIZE",
                          os.environ.get("SLURM_NTASKS_PER_NODE", "4")))
    my_node = rank // ranks_per_node
    intra_halo_size = 0
    inter_halo_size = 0
    for r, verts in enumerate(recv_by_rank):
        peer_node = r // ranks_per_node
        if peer_node == my_node:
            intra_halo_size += len(verts)
        else:
            inter_halo_size += len(verts)

    return {
        "local_vertices": local_vertices,
        "n_local": n_local,
        "n_halo": len(halo_order),
        "edge_index": edge_index,
        "send_counts": send_counts,
        "recv_counts": recv_counts,
        "send_idx_by_rank": send_idx_by_rank,
        "halo_order": halo_order,
        "intra_halo_size": intra_halo_size,
        "inter_halo_size": inter_halo_size,
        "ranks_per_node": ranks_per_node,
    }


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
        recv_list = [torch.zeros(rc, x_local.shape[1],
                                  dtype=x_local.dtype, device=x_local.device)
                     for rc in recv_counts]

        dist.all_to_all(recv_list, send_list)

        recv_buf = torch.cat(recv_list, dim=0) if sum(recv_counts) > 0 else \
            torch.zeros(0, x_local.shape[1], device=x_local.device)

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
        send_idx_flat, = ctx.saved_tensors
        send_counts = ctx.send_counts
        recv_counts = ctx.recv_counts
        world_size = ctx.world_size
        n_local = ctx.n_local
        F = ctx.feature_dim
        device = ctx.device

        # Reverse: recv_counts become send_counts and vice versa
        grad_recv_list = list(grad_recv.split(recv_counts, dim=0)) \
            if grad_recv.shape[0] > 0 else [torch.zeros(0, F, device=device)] * world_size
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
# GNN layers (same as bench_compute.py for consistency)
# ===========================================================================

class GCNLayer(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, feature_dim, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        n_local = x.shape[0]
        # Only update local vertices (dst < n_local guard not needed since
        # edge_index already restricts to local dst)
        msg = self.linear(x[src])
        out = torch.zeros_like(x)
        out.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
        return out


class EdgeConditionedLayer(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        msg = self.mlp(torch.cat([x[src], x[dst], edge_attr], dim=-1))
        out = torch.zeros_like(x)
        out.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
        return out


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="End-to-end halo exchange benchmark")
    p.add_argument("--graph", choices=["erdos_renyi", "sbm"], default="erdos_renyi")
    p.add_argument("--num-vertices", type=int, default=100_000)
    p.add_argument("--avg-degree", type=float, default=20.0)
    p.add_argument("--sbm-inter-density", type=float, default=0.1,
                   help="Fraction of inter-block edges for SBM graphs")
    p.add_argument("--feature-dim", type=int, default=128)
    p.add_argument("--model", choices=["gcn", "edge"], default="gcn")
    p.add_argument("--partitioner", choices=["random", "balanced", "metis"],
                   default="balanced")
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
        edges = gen_sbm(args.num_vertices, args.avg_degree,
                        args.sbm_inter_density, rng)

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
    send_idx_flat = torch.cat([
        torch.tensor(s, dtype=torch.long) for s in pattern["send_idx_by_rank"]
    ]).to(device) if sum(send_counts) > 0 else torch.zeros(0, dtype=torch.long, device=device)

    # --- Model ---
    if args.model == "gcn":
        layer = GCNLayer(F).to(device)
    else:
        layer = EdgeConditionedLayer(F).to(device)
    layer.train()

    # --- Synthetic local node features ---
    x_local = torch.randn(n_local, F, device=device, requires_grad=True)
    edge_attr = torch.randn(edge_index.shape[1], F, device=device) \
        if args.model == "edge" else None

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
