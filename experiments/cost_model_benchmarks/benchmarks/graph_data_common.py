import os

import numpy as np
import torch
import torch.distributed as dist


# ===========================================================================
# Synthetic graph generators
# ===========================================================================


def gen_erdos_renyi(
    num_vertices: int, avg_degree: float, rng: np.random.Generator
) -> np.ndarray:
    """Return edge array of shape [E, 2] (src, dst) for an Erdős-Rényi digraph."""
    num_edges = int(num_vertices * avg_degree)
    src = rng.integers(0, num_vertices, size=num_edges)
    dst = rng.integers(0, num_vertices, size=num_edges)
    return np.stack([src, dst], axis=1)


def gen_sbm(
    num_vertices: int, avg_degree: float, inter_density: float, rng: np.random.Generator
) -> np.ndarray:
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
    d_block = (
        s // block_size + 1 + rng.integers(0, world_size - 1, size=inter_edges)
    ) % world_size
    d = d_block * block_size + rng.integers(0, block_size, size=inter_edges)
    d = np.clip(d, 0, num_vertices - 1)
    edges.append(np.stack([s, d], axis=1))

    return np.concatenate(edges, axis=0)


# ===========================================================================
# Partitioners
# ===========================================================================


def partition_random(
    num_vertices: int, world_size: int, rng: np.random.Generator
) -> np.ndarray:
    return rng.integers(0, world_size, size=num_vertices).astype(np.int64)


def partition_balanced(num_vertices: int, world_size: int) -> np.ndarray:
    return np.floor(np.arange(num_vertices) * world_size / num_vertices).astype(
        np.int64
    )


def partition_metis(
    num_vertices: int, world_size: int, edges: np.ndarray
) -> np.ndarray:
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
# Communication-pattern derived stats
#
# The comm pattern itself (local vertex/edge remapping, send/recv CSR
# indexing, per-rank comm map) is built exclusively via
# ``DGraph.distributed.build_communication_pattern`` — see bench_crossover.py
# and bench_end_to_end.py.  This module previously hand-rolled its own
# ``build_local_comm_pattern`` with an independent (and inconsistent) halo/
# send/recv layout that was never a ``CommunicationPattern`` and was not
# ordered per-rank the way ``recv_offset``'s CSR layout requires, which made
# it unusable as input to ``DGraph.distributed.HaloExchange``. Only the
# small derived-stats helpers below remain; they consume a real
# ``CommunicationPattern``.
# ===========================================================================


def get_ranks_per_node() -> int:
    """Return the number of ranks co-located on this node.

    Required to split a comm pattern's halo traffic into intra-/inter-node
    volumes (``c_intra_bytes`` vs ``c_inter_bytes``) for the hierarchical
    cost model. Refuses to guess: silently defaulting this would bias every
    downstream fit.
    """
    ranks_per_node_str = os.environ.get(
        "LOCAL_WORLD_SIZE", os.environ.get("SLURM_NTASKS_PER_NODE")
    )
    if ranks_per_node_str is None:
        raise RuntimeError(
            "Neither LOCAL_WORLD_SIZE nor SLURM_NTASKS_PER_NODE is set — cannot "
            "determine GPUs-per-node. This value directly determines the "
            "intra/inter halo split (c_intra_bytes vs c_inter_bytes) that the "
            "hierarchical cost model relies on, so silently guessing would "
            "bias every downstream fit. Launch with torchrun (which sets "
            "LOCAL_WORLD_SIZE) or under srun/SLURM (which sets "
            "SLURM_NTASKS_PER_NODE)."
        )
    return int(ranks_per_node_str)


def intra_inter_halo(comm_pattern, ranks_per_node: int) -> tuple:
    """Split a ``CommunicationPattern``'s halo receive counts into
    (intra_node, inter_node) vertex totals, using ``recv_offset`` — the
    authoritative per-source-rank CSR layout ``build_communication_pattern``
    produces — rather than re-deriving halo ownership by hand.
    """
    rank = comm_pattern.rank
    my_node = rank // ranks_per_node
    recv_counts = (
        comm_pattern.recv_offset[1:] - comm_pattern.recv_offset[:-1]
    ).tolist()
    intra = 0
    inter = 0
    for r, count in enumerate(recv_counts):
        if r == rank:
            continue
        if (r // ranks_per_node) == my_node:
            intra += int(count)
        else:
            inter += int(count)
    return intra, inter
