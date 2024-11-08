# Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LBANN Research Team (B. Van Essen, et al.) listed in
# the CONTRIBUTORS file. See the top-level LICENSE file for details.
#
# LLNL-CODE-697807.
# All rights reserved.
#
# This file is part of LBANN: Livermore Big Artificial Neural Network
# Toolkit. For details, see http://software.llnl.gov/LBANN or
# https://github.com/LBANN and https://github.com/LLNL/LBANN.
#
# SPDX-License-Identifier: (Apache-2.0)
import torch.distributed as dist
from DGraph.utils import largest_split
from torch import Tensor
from typing import List
import torch


def _nccl_gather_op(send_tensor_buffer, recv_tensor_buffer, indices, rank, world_size):
    p2p_op_list = []
    mb_size = send_tensor_buffer.shape[0]
    num_input_rows = largest_split(send_tensor_buffer.shape[1], world_size)
    num_output_rows = largest_split(recv_tensor_buffer.shape[1], world_size)

    for mb in range(mb_size):
        for i, ind_i in enumerate(indices[mb]):
            src_rank = ind_i // num_input_rows
            recv_rank = i // num_output_rows
            send_tensor = send_tensor_buffer[mb, ind_i % num_input_rows]
            recv_tensor = recv_tensor_buffer[mb, i % num_output_rows]

            if src_rank == recv_rank:
                continue

            if rank == src_rank:
                p2p_op_list.append(dist.P2POp(dist.isend, send_tensor, recv_rank))

            if rank == recv_tensor:
                p2p_op_list.append(dist.P2POp(dist.irecv, recv_tensor, src_rank))
    reqs = dist.batch_isend_irecv(p2p_op_list)

    for req in reqs:
        req.wait()


def _optimized_nccl_gather_op(
    send_buffer: Tensor,
    recv_buffer_list: List[Tensor],
    send_ranks: Tensor,
    recv_ranks: Tensor,
    recv_comm_vector: Tensor,
    rank: int,
    world_size: int,
) -> List[Tensor]:
    """
    An optimized version of the gather operation that uses NCCL to gather data
    from multiple ranks. It implements a vector all-gather operation where each rank
    sends a different buffer to each rank depending on the indices.

    This ensures that for any data, only one coalesced message is sent between
    each pair of ranks.

    Note: The receving rank is in-charge of placing the data in the correct
    position in the output tensor.

    Args:

    """

    p2p_op_list = []

    assert (
        send_ranks != recv_ranks
    ).any(), "No self-sends allowed. Should be done in the local gather."

    for send_rank_index in range(world_size):
        for recv_rank_index in range(world_size):
            if send_rank_index == recv_rank_index:
                # No self-sends allowed. Should be done in the local gather.
                continue
            if (send_rank_index != rank) and (recv_rank_index != rank):
                # Current rank not involved in this p2p communication pair.
                continue

            if recv_comm_vector[send_rank_index] == 0:
                # No communication between these ranks.
                continue
            # Current rank is involved in this p2p communication pair.
            # Also there is communication between these ranks according to
            # the communication matrix.
            if send_rank_index == rank:
                # Current rank is sending data to recv_rank_index.
                send_tensor = send_buffer
                p2p_op_list.append(dist.P2POp(dist.isend, send_tensor, recv_rank_index))

            if recv_rank_index == rank:
                # Current rank is receiving data from send_rank_index.
                recv_tensor = recv_buffer_list[send_rank_index]
                p2p_op_list.append(dist.P2POp(dist.irecv, recv_tensor, send_rank_index))

    reqs = dist.batch_isend_irecv(p2p_op_list)

    for req in reqs:
        req.wait()

    torch.cuda.synchronize()
    return recv_buffer_list
