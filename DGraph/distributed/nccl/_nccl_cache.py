from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
from DGraph.distributed.RankLocalOps import RankLocalRenumberingWithMapping
from DGraph.distributed.nccl._indices_utils import (
    _get_local_recv_placement,
    _get_local_send_placement,
    _get_local_unique_recv_placement,
    _get_recv_comm_vector,
    _get_send_comm_vector,
    _get_send_recv_comm_vectors,
)
import torch.distributed as dist


@dataclass
class NCCLGatherCache:
    """This class stores the NCCL communication cache required for alltoallv operations
    for a gather operation.
    """

    # Forward cached values
    gather_recv_comm_vector: torch.Tensor
    gather_send_comm_vector: torch.Tensor
    gather_recv_local_placement: Dict[int, torch.Tensor]
    gather_send_local_placement: Dict[int, torch.Tensor]

    # Backward cached values
    scatter_renumbered_indices: torch.Tensor
    scatter_local_comm_mask: torch.Tensor
    scatter_remote_send_to_ranks: torch.Tensor
    scatter_num_remote_rows: int
    scatter_recv_local_placement: Dict[int, torch.Tensor]
    scatter_local_remapped_ranks: torch.Tensor
    rank: int
    world_size: int


@dataclass
class NCCLScatterCache:
    """This class stores the NCCL communication cache required for alltoallv operations
    for a scatter operation.
    """

    # Forward cached values
    scatter_recv_local_placement: Dict[int, torch.Tensor]
    scatter_local_comm_mask: torch.Tensor
    scatter_remote_send_to_ranks: torch.Tensor
    scatter_num_remote_rows: int
    scatter_local_remapped_ranks: torch.Tensor
    scatter_local_renumbered_indices: torch.Tensor
    # Backward cached values
    gather_num_output_rows: int
    gather_recv_comm_vector: torch.Tensor
    gather_send_comm_vector: torch.Tensor
    gather_recv_local_placement: Dict[int, torch.Tensor]
    gather_send_local_placement: Dict[int, torch.Tensor]
    rank: int
    world_size: int


def all_to_all_cache_helper(
    indices, edge_placement, edge_vertex_ranks, num_rows, rank, world_size
):
    local_mask = edge_placement == rank
    all_comm_mask = edge_placement != edge_vertex_ranks
    comm_senders = edge_placement[all_comm_mask]
    comm_receivers = edge_vertex_ranks[all_comm_mask]

    local_vertex_src_ranks = edge_vertex_ranks[local_mask]

    send_to_ranks = comm_receivers[comm_senders == rank]
    receive_from_ranks = comm_senders[comm_receivers == rank]

    send_comm_vector = torch.bincount(send_to_ranks, minlength=world_size).long()
    recv_comm_vector = torch.bincount(receive_from_ranks, minlength=world_size).long()

    recv_local_placement = {}

    for i, num_messages in enumerate(recv_comm_vector):
        if num_messages == 0:
            continue

        if i == rank:
            continue

        _local_placement_indices = torch.argwhere(local_vertex_src_ranks == i)
        recv_local_placement[i] = _local_placement_indices

    send_local_placement = {}
    for i, num_messages in enumerate(send_comm_vector):
        if num_messages == 0:
            # Not sending any messages current_rank to rank i
            continue
        if i == rank:
            # No local sends
            continue

        _mask = (edge_placement == rank) & (edge_vertex_ranks == i)
        _send_row = indices[0][_mask] % num_rows
        send_local_placement[i] = _send_row

    return (
        send_comm_vector,
        recv_comm_vector,
        send_local_placement,
        recv_local_placement,
    )


def all_to_all_cache_with_local_reduce_helper(
    indices, edge_placement, edge_vertex_ranks, num_output_rows, rank, world_size
):
    indices = indices.view(-1)
    all_comm_mask = edge_placement != edge_vertex_ranks
    receiver_mask = edge_vertex_ranks == rank

    # All message that will be recieved by the current rank but
    # sent by other ranks

    local_edges_mask = edge_placement == rank
    local_indices_slice = indices[local_edges_mask]
    local_dest_ranks_slice = edge_vertex_ranks[local_edges_mask]

    # This is the mask for the rows that will be sent by the current rank
    local_send_mask = local_dest_ranks_slice != rank

    local_remote_send_indices = local_indices_slice[local_send_mask]
    local_remote_dest_mappings = local_dest_ranks_slice[local_send_mask]

    renumbered_indices, unique_indices, remapped_ranks = (
        RankLocalRenumberingWithMapping(
            local_remote_send_indices, local_remote_dest_mappings
        )
    )
    receving_ranks = torch.unique(local_dest_ranks_slice[local_send_mask])

    # All message that will be recieved by the current rank but
    remote_recv_mask = all_comm_mask & receiver_mask
    recv_placement = _get_local_unique_recv_placement(
        indices, edge_placement, remote_recv_mask, num_output_rows, rank, world_size
    )
    num_remote_rows = len(unique_indices)

    return (
        renumbered_indices,
        num_remote_rows,
        remapped_ranks,
        receving_ranks,
        recv_placement,
    )


def NCCLScatterCacheGenerator(
    indices: torch.Tensor,
    edge_placement: torch.Tensor,
    edge_dest_ranks: torch.Tensor,
    num_output_rows: int,
    rank: int,
    world_size: int,
) -> NCCLScatterCache:
    """
    This function generates the NCCL cache required for alltoallv operations.
    """

    # information for the forward pass
    all_comm_mask = edge_placement != edge_dest_ranks
    receiver_mask = edge_dest_ranks == rank
    remote_recv_mask = all_comm_mask & receiver_mask

    # All message that will be recieved by the current rank but
    # sent by other ranks
    indices = indices.view(-1)
    local_edges_mask = edge_placement == rank
    local_indices_slice = indices[local_edges_mask]
    local_dest_ranks_slice = edge_dest_ranks[local_edges_mask]

    # This is the mask for the rows that will be sent by the current rank
    local_send_mask = local_dest_ranks_slice != rank

    local_remote_send_indices = local_indices_slice[local_send_mask]
    local_remote_dest_mappings = local_dest_ranks_slice[local_send_mask]

    renumbered_indices, unique_indices, remapped_ranks = (
        RankLocalRenumberingWithMapping(
            local_remote_send_indices, local_remote_dest_mappings
        )
    )

    num_remote_rows = unique_indices.shape[0]

    receving_ranks = torch.unique(local_dest_ranks_slice[local_send_mask])

    recv_placement = _get_local_unique_recv_placement(
        indices, edge_placement, remote_recv_mask, num_output_rows, rank, world_size
    )

    # Information for the backward pass
    # It's a gather operation so quite a bit simpler

    num_grad_output_rows = int(local_edges_mask.sum().item())
    send_comm_vector, recv_comm_vector, send_local_placement, recv_local_placement = (
        all_to_all_cache_helper(
            indices,
            edge_placement,
            edge_dest_ranks,
            num_grad_output_rows,
            rank,
            world_size,
        )
    )

    _cache = NCCLScatterCache(
        scatter_recv_local_placement=recv_placement,
        scatter_local_comm_mask=local_send_mask,
        scatter_remote_send_to_ranks=receving_ranks,
        scatter_num_remote_rows=num_remote_rows,
        scatter_local_remapped_ranks=remapped_ranks,
        scatter_local_renumbered_indices=renumbered_indices,
        gather_num_output_rows=num_grad_output_rows,
        gather_recv_comm_vector=recv_comm_vector,
        gather_send_comm_vector=send_comm_vector,
        gather_recv_local_placement=recv_local_placement,
        gather_send_local_placement=send_local_placement,
        rank=rank,
        world_size=world_size,
    )
    return _cache


def NCCLGatherCacheGenerator(
    indices: torch.Tensor,
    edge_placement: torch.Tensor,
    edge_dest_ranks: torch.Tensor,
    num_input_rows: int,
    rank: int,
    world_size: int,
):
    """
    This function generates the NCCL cache required for alltoallv operations.
    """

    # Forward pass

    send_comm_vector, recv_comm_vector, send_local_placement, recv_local_placement = (
        all_to_all_cache_helper(
            indices, edge_placement, edge_dest_ranks, num_input_rows, rank, world_size
        )
    )

    local_slice_mask = edge_placement == rank
    local_dest_ranks = edge_dest_ranks[local_slice_mask]

    local_send_mask = local_dest_ranks != rank

    # Backward pass
    # This is a scatter operation so quite a bit more complicated

    all_to_all_v_cache = all_to_all_cache_with_local_reduce_helper(
        indices,
        edge_placement,
        edge_dest_ranks,
        num_input_rows,
        rank,
        world_size,
    )

    (
        renumbered_indices,
        num_remote_rows,
        remapped_ranks,
        receving_ranks,
        scatter_recv_local_placement,
    ) = all_to_all_v_cache

    _cache = NCCLGatherCache(
        gather_recv_comm_vector=recv_comm_vector,
        gather_send_comm_vector=send_comm_vector,
        gather_recv_local_placement=recv_local_placement,
        gather_send_local_placement=send_local_placement,
        scatter_renumbered_indices=renumbered_indices,
        scatter_local_comm_mask=local_send_mask,
        scatter_remote_send_to_ranks=receving_ranks,
        scatter_num_remote_rows=num_remote_rows,
        scatter_recv_local_placement=scatter_recv_local_placement,
        scatter_local_remapped_ranks=remapped_ranks,
        rank=rank,
        world_size=world_size,
    )
    return _cache
