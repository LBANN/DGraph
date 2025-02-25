from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
from DGraph.distributed.RankLocalOps import RankLocalRenumberingWithMapping
from DGraph.distributed.nccl._indices_utils import (
    _get_local_recv_placement,
    _get_local_send_placement,
    _get_recv_comm_vector,
    _get_send_comm_vector,
    _get_send_recv_comm_vectors,
)


@dataclass
class NCCLGatherCache:
    """This class stores the NCCL communication cache required for alltoallv operations
    for a gather operation.
    """

    send_buffer_dict: Dict[int, torch.Tensor]
    recv_buffer_dict: Dict[int, torch.Tensor]
    send_comm_vector: torch.Tensor
    recv_comm_vector: torch.Tensor
    send_local_placement: torch.Tensor
    recv_local_placement: torch.Tensor

    backward_renumbered_indices: torch.Tensor
    backward_num_remote_rows: int
    backward_recv_placement: Dict[int, torch.Tensor]
    rank: int
    world_size: int
    num_features: int


@dataclass
class NCCLScatterCache:
    """This class stores the NCCL communication cache required for alltoallv operations
    for a scatter operation.
    """

    send_comm_vector: torch.Tensor
    recv_comm_vector: torch.Tensor

    local_comm_mask: torch.Tensor
    send_local_placement: Dict[int, torch.Tensor]
    recv_local_placement: Dict[int, torch.Tensor]
    num_remote_rows: int
    local_remapped_ranks: torch.Tensor
    local_renumbered_indices: torch.Tensor
    rank: int
    world_size: int


def NCCLScatterCacheGenerator(
    indices: torch.Tensor,
    src_ranks: torch.Tensor,
    dest_ranks: torch.Tensor,
    num_output_rows: int,
    rank: int,
    world_size: int,
) -> NCCLScatterCache:
    """
    This function generates the NCCL cache required for alltoallv operations.
    """

    all_comm_mask = src_ranks != dest_ranks
    receiver_mask = dest_ranks == rank

    # All message that will be recieved by the current rank but
    # sent by other ranks

    receive_from_ranks = all_comm_mask & receiver_mask

    _start_index = ((indices.shape[0] + world_size - 1) // world_size) * rank
    _end_index = ((indices.shape[0] + world_size - 1) // world_size) * (rank + 1)
    _end_index = min(_end_index, indices.shape[0])

    local_indices_slice = indices[_start_index:_end_index]
    local_dest_ranks_slice = dest_ranks[_start_index:_end_index]

    # This is the mask for the rows that will be sent by the current rank
    local_send_mask = local_dest_ranks_slice != rank

    send_comm_vector, recv_comm_vector = _get_send_recv_comm_vectors(
        src_ranks, dest_ranks, rank, world_size
    )
    send_local_placement = _get_local_send_placement(
        send_comm_vector, indices, src_ranks, dest_ranks, rank, num_output_rows
    )

    recv_local_placement = _get_local_recv_placement(recv_comm_vector, src_ranks, rank)

    local_comm_indices = local_indices_slice[local_send_mask]
    local_remote_dest_mappings = local_dest_ranks_slice[local_send_mask]

    renumbered_indices, unique_indices, remapped_ranks = (
        RankLocalRenumberingWithMapping(local_comm_indices, local_remote_dest_mappings)
    )
    num_remote_rows = unique_indices.shape[0]
    _cache = NCCLScatterCache(
        local_comm_mask=local_send_mask,
        send_local_placement=send_local_placement,
        recv_local_placement=recv_local_placement,
        num_remote_rows=num_remote_rows,
        local_remapped_ranks=remapped_ranks,
        local_renumbered_indices=renumbered_indices,
        rank=rank,
        world_size=world_size,
    )
    return _cache


def NCCLGatherCacheGenerator(
    indices: torch.Tensor,
    edge_src_ranks: torch.Tensor,
    edge_dest_ranks: torch.Tensor,
    rank: int,
    world_size: int,
):
    """
    This function generates the NCCL cache required for alltoallv operations.
    """

    all_comm_mask = edge_src_ranks != edge_dest_ranks
    receiver_mask = edge_dest_ranks == rank

    # All message that will be recieved by the current rank but
    # sent by other ranks

    receive_from_ranks = all_comm_mask & receiver_mask

    local_indices_mask = edge_src_ranks == rank
    local_indices_slice = indices[local_indices_mask]
    local_dest_ranks_slice = edge_dest_ranks[local_indices_mask]

    # This is the mask for the rows that will be sent by the current rank
    local_send_mask = local_dest_ranks_slice != rank

    # send_comm_vector recv_comm_vector are the number of messages
    # to be sent to each rank and received from each rank respectively
    # Shape: (world_size,)
    send_comm_vector, recv_comm_vector = _get_send_recv_comm_vectors(
        edge_src_ranks, edge_dest_ranks, rank, world_size
    )

    send_local_placement = _get_local_send_placement(
        send_comm_vector,
        indices,
        edge_src_ranks,
        edge_dest_ranks,
        rank,
        num_output_rows,
    )

    recv_local_placement = _get_local_recv_placement(
        recv_comm_vector, edge_src_ranks, rank
    )

    local_comm_indices = local_indices_slice[local_send_mask]
    local_remote_dest_mappings = local_dest_ranks_slice[local_send_mask]

    renumbered_indices, unique_indices, remapped_ranks = (
        RankLocalRenumberingWithMapping(local_comm_indices, local_remote_dest_mappings)
    )
    num_remote_rows = unique_indices.shape[0]
    _cache = NCCLGatherCache(
        send_buffer_dict={},
        recv_buffer_dict={},
        send_comm_vector=send_comm_vector,
        recv_comm_vector=recv_comm_vector,
        send_local_placement=send_local_placement,
        recv_local_placement=recv_local_placement,
        rank=rank,
        world_size=world_size,
        num_features=num_features,
    )
    return _cache
