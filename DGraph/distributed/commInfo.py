import torch
from dataclasses import dataclass
from typing import Optional
import torch.distributed as dist


@dataclass
class CommunicationPattern:
    # --- Identity ---
    rank: int
    world_size: int

    # --- Vertex Counts ---
    num_local_vertices: int
    num_halo_vertices: int

    # --- Local Subgraph ---
    local_edge_list: torch.Tensor  # [num_local_edges, 2]

    # --- Send Indexing ---
    send_local_idx: torch.Tensor  # [total_sends]
    send_offset: torch.Tensor  # [world_size + 1]

    # --- Receive Indexing ---
    recv_offset: torch.Tensor  # [world_size + 1]

    # --- Communication Map ---
    comm_map: torch.Tensor  # [world_size, world_size]

    # --- One-Sided RMA Offsets ---
    put_forward_remote_offset: torch.Tensor  # [world_size]
    put_backward_remote_offset: torch.Tensor  # [world_size]


def compute_local_vertices(partitioning: torch.Tensor, rank: int) -> torch.Tensor:
    """Returns local_vertices_global: [num_local] global IDs owned by this rank"""

    return torch.where(partitioning == rank)[0]


def compute_halo_vertices(
    edge_list: torch.Tensor,
    src_partitioning: torch.Tensor,
    rank: int,
    dst_partitioning: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes halo vertices. Supports both homogeneous and bipartite/heterogeneous relations.
    """
    # Fallback for homogeneous graphs
    if dst_partitioning is None:
        dst_partitioning = src_partitioning

    src_rank = src_partitioning[edge_list[:, 0]]
    dst_rank = dst_partitioning[edge_list[:, 1]]

    # Cross-rank mask: source is local, destination is remote
    cross_mask = (src_rank == rank) & (dst_rank != rank)

    # Return unique destination vertex IDs from those edges
    return torch.unique(edge_list[cross_mask, 1])


def compute_local_edge_list(
    global_edge_list: torch.Tensor,  # [E, 2]
    partitioning: torch.Tensor,  # [V]
    local_vertices_global: torch.Tensor,  # [num_local]
    halo_vertices_global: torch.Tensor,  # [num_halo]
    rank: int,
) -> torch.Tensor:
    num_local = local_vertices_global.size(0)
    num_halo = halo_vertices_global.size(0)
    num_global = partitioning.size(0)

    # Filter edges owned by this rank
    local_edge_mask = partitioning[global_edge_list[:, 0]] == rank
    local_edges_global = global_edge_list[local_edge_mask]

    # Build inverse map: global_id -> local_idx via scatter
    g2l = torch.empty(num_global, dtype=torch.long, device=global_edge_list.device)
    g2l.scatter_(0, local_vertices_global, torch.arange(num_local, device=g2l.device))
    g2l.scatter_(
        0,
        halo_vertices_global,
        torch.arange(num_local, num_local + num_halo, device=g2l.device),
    )

    # Remap to local numbering
    local_edge_list = g2l[local_edges_global]

    return local_edge_list


def compute_boundary_vertices(
    edge_list: torch.Tensor,
    src_partitioning: torch.Tensor,
    src_local_vertices_global: torch.Tensor,
    rank: int,
    num_ranks: int,
    dst_partitioning: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes boundary vertices and CSR send offsets.
    """
    # Fallback for homogeneous graphs
    if dst_partitioning is None:
        dst_partitioning = src_partitioning

    # 1. Filter cross-rank edges
    src_rank = src_partitioning[edge_list[:, 0]]
    dst_rank = dst_partitioning[edge_list[:, 1]]
    cross_mask = (src_rank == rank) & (dst_rank != rank)
    cross_edges = edge_list[cross_mask]

    # 2. Deduplicate (src, dst_rank) pairs
    src_global = cross_edges[:, 0]
    target_ranks = dst_partitioning[cross_edges[:, 1]]

    # v_src_total acts as V for homogeneous, or V_src for heterogeneous
    v_src_total = src_partitioning.size(0)
    encoded = target_ranks * v_src_total + src_global
    unique_encoded = torch.unique(encoded)

    target_ranks_unique = unique_encoded // v_src_total
    src_global_unique = unique_encoded % v_src_total

    # 3. Sort by target rank
    sort_idx = torch.argsort(target_ranks_unique)
    target_ranks_sorted = target_ranks_unique[sort_idx]
    src_global_sorted = src_global_unique[sort_idx]

    # 4. Remap to local indices
    num_local = src_local_vertices_global.size(0)
    g2l = torch.empty(v_src_total, dtype=torch.long, device=edge_list.device)
    g2l[src_local_vertices_global] = torch.arange(num_local, device=edge_list.device)
    send_local_idx = g2l[src_global_sorted]

    # 5. Build CSR offsets
    send_offset = torch.zeros(num_ranks + 1, dtype=torch.long, device=edge_list.device)
    send_offset.scatter_add_(
        0, target_ranks_sorted + 1, torch.ones_like(target_ranks_sorted)
    )
    send_offset = send_offset.cumsum(0)

    return send_local_idx, send_offset


def compute_comm_map(send_offset, world_size) -> torch.Tensor:
    """All-gathers send counts to build comm_map: [world_size, world_size]"""
    send_counts = send_offset[1:] - send_offset[:-1]
    comm_map_list = [torch.zeros(world_size).long().cuda() for _ in range(world_size)]
    dist.all_gather(comm_map_list, send_counts.cuda())
    comm_map = torch.stack(comm_map_list)
    return comm_map


def compute_recv_offsets(comm_map, rank) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (recv_offset, recv_backward_offset) from comm_map"""
    recv_counts = comm_map[:, rank]
    recv_offset = torch.zeros(comm_map.size(0) + 1, dtype=torch.long)
    recv_offset[1:] = recv_counts.cpu().cumsum(0)

    recv_backward_offset = comm_map[:rank, :].sum(0)
    return recv_offset, recv_backward_offset


def build_communication_pattern(
    global_edge_list: torch.Tensor,
    partitioning: torch.Tensor,
    rank: int,
    world_size: int,
) -> CommunicationPattern:
    """

    Args:
        global_edge_list (torch.Tensor)): A tensor of shape [E, 2]
        partitioning (torch.Tensor): A tensor of shape [V]
        rank (int): Rank of this process
        world_size (int): Total number of processes

    Returns:
        CommunicationPattern
    """
    local_verts = compute_local_vertices(partitioning, rank)
    halo_verts = compute_halo_vertices(global_edge_list, partitioning, rank)
    local_edges = compute_local_edge_list(
        global_edge_list, partitioning, local_verts, halo_verts, rank
    )
    send_idx, send_off = compute_boundary_vertices(
        global_edge_list, partitioning, local_verts, rank, world_size
    )
    comm = compute_comm_map(send_off, world_size)
    recv_off, recv_back_off = compute_recv_offsets(comm, rank)

    return CommunicationPattern(
        rank=rank,
        world_size=world_size,
        num_local_vertices=local_verts.size(0),
        num_halo_vertices=halo_verts.size(0),
        comm_map=comm,
        send_local_idx=send_idx,
        send_offset=send_off,
        recv_offset=recv_off,
        local_edge_list=local_edges,
        put_forward_remote_offset=comm[:rank, :].sum(0),
        put_backward_remote_offset=comm[:, :rank].sum(1),
    )
