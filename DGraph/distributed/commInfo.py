import torch
from dataclasses import dataclass
from typing import Tuple
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


def compute_halo_vertices(global_edge_list, partitioning, rank) -> torch.Tensor:
    """Returns halo_vertices_global: [num_halo] global IDs of remote vertices
    that share an edge with a local vertex"""
    src_rank = partitioning[global_edge_list[:, 0]]
    dst_rank = partitioning[global_edge_list[:, 1]]
    cross_mask = (src_rank == rank) & (dst_rank != rank)
    return torch.unique(global_edge_list[cross_mask, 1])


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
    global_edge_list: torch.Tensor,  # [E, 2]
    partitioning: torch.Tensor,  # [V]
    local_vertices_global: torch.Tensor,  # [num_local]
    rank: int,
    num_ranks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        send_local_idx: [num_sends] local indices into local_buffer
        send_offset:    [num_ranks + 1] such that send_local_idx[send_offset[i]:send_offset[i+1]]
                        are the local vertices to send to rank i
    """
    # Filter edges where source is local, destination is remote
    src_rank = partitioning[global_edge_list[:, 0]]
    dst_rank = partitioning[global_edge_list[:, 1]]
    cross_mask = (src_rank == rank) & (dst_rank != rank)
    cross_edges = global_edge_list[cross_mask]

    # For each cross edge, we have (local_src_global_id, remote_dst_global_id)
    # We need unique (src, dst_rank) pairs — a vertex may have multiple edges
    # to the same remote rank, but we only send it once per rank
    src_global = cross_edges[:, 0]
    target_ranks = partitioning[cross_edges[:, 1]]

    # Build (target_rank, src_global) pairs and deduplicate
    # Encode as a single int for unique: target_rank * V + src_global
    num_vertices = partitioning.size(0)
    encoded = target_ranks * num_vertices + src_global
    unique_encoded = torch.unique(encoded)

    target_ranks_unique = unique_encoded.div(num_vertices, rounding_mode="floor")
    src_global_unique = unique_encoded % num_vertices

    # Sort by target rank to get contiguous grouping
    sort_idx = torch.argsort(target_ranks_unique)
    target_ranks_sorted = target_ranks_unique[sort_idx]
    src_global_sorted = src_global_unique[sort_idx]

    # Build inverse map: global -> local index for local vertices only
    g2l = torch.empty(num_vertices, dtype=torch.long, device=global_edge_list.device)
    g2l[local_vertices_global] = torch.arange(
        local_vertices_global.size(0), device=g2l.device
    )

    # Remap to local indices
    send_local_idx = g2l[src_global_sorted]

    # Compute send_offset from the sorted target ranks
    send_offset = torch.zeros(
        num_ranks + 1, dtype=torch.long, device=global_edge_list.device
    )
    ones = torch.ones_like(target_ranks_sorted)
    send_offset.scatter_add_(0, target_ranks_sorted + 1, ones)
    send_offset = send_offset.cumsum(0)

    return send_local_idx, send_offset


def compute_comm_map(send_offset, world_size) -> torch.Tensor:
    """All-gathers send counts to build comm_map: [world_size, world_size]"""
    send_counts = send_offset[1:] - send_offset[:-1]
    comm_map_list = [torch.zeros(world_size).cuda() for _ in range(world_size)]
    dist.all_gather(comm_map_list, send_counts)
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
