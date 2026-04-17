import numpy as np
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
# Minimal halo-exchange infrastructure
# ===========================================================================


def build_local_comm_pattern(
    edges: np.ndarray, assignment: np.ndarray, rank: int, world_size: int
):
    """Compute the local communication pattern for this rank.

    Returns a CommunicationPattern object with:
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
        owned_mask = (
            assignment[needed] == rank if len(needed) > 0 else np.array([], dtype=bool)
        )
        owned = needed[owned_mask] if len(needed) > 0 else np.array([], dtype=np.int64)
        # Map to local indices
        local_idxs = np.array([g2l[int(v)] for v in owned], dtype=np.int64)
        send_idx_by_rank.append(local_idxs)

    send_counts = [len(s) for s in send_idx_by_rank]

    # Remap edges to local indices
    valid_edge_mask = np.array(
        [(int(s) in all_g2l) and (int(d) in all_g2l) for s, d in local_edges]
    )
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
    ranks_per_node = int(
        os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("SLURM_NTASKS_PER_NODE", "4"))
    )
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
