from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
from DGraph.distributed.nccl._indices_utils import (
    _get_local_recv_placement,
    _get_local_send_placement,
    _get_recv_comm_vector,
    _get_send_comm_vector,
    _get_send_recv_comm_vectors,
)


@dataclass
class NCCLGatherCache:
    """A class to store the NCCL communication cache required for alltoallv operations
    for a gather operation.
    """

    send_buffer_dict: Dict[int, torch.Tensor]
    recv_buffer_dict: Dict[int, torch.Tensor]
    send_comm_vector: torch.Tensor
    recv_comm_vector: torch.Tensor
    send_local_placement: torch.Tensor
    recv_local_placement: torch.Tensor
    rank: int
    world_size: int
    num_features: int


@dataclass
class NCCLScatterCache:
    """A class to store the NCCL communication cache required for alltoallv operations
    for a scatter operation.
    """

    local_comm_mask: torch.Tensor
    send_local_placement: torch.Tensor
    recv_local_placement: Dict[int, torch.Tensor]
    num_remote_rows: int
    local_remapped_ranks: torch.Tensor
    local_renumbered_indices: torch.Tensor
    rank: int
    world_size: int
    num_features: int


def NCCLCacheGenerator(
    indices: torch.Tensor,
    src_ranks: torch.Tensor,
    dest_ranks: torch.Tensor,
    num_output_rows: int,
    rank: int,
    world_size: int,
):
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
