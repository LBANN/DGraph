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


def _nccl_scatter_op(send_tensor_buffer, recv_tensor_buffer, indices, rank, world_size):
    p2p_op_list = []
    mb_size = send_tensor_buffer.shape[0]
    num_input_rows = largest_split(send_tensor_buffer.shape[1], world_size)
    num_output_rows = largest_split(recv_tensor_buffer.shape[1], world_size)

    for mb in range(mb_size):
        for i, ind_i in enumerate(indices[mb]):
            src_rank = i // num_input_rows
            recv_rank = ind_i // num_output_rows
            send_tensor = send_tensor_buffer[mb, i % num_input_rows]
            recv_tensor = recv_tensor_buffer[mb, ind_i % num_output_rows]

            if src_rank == recv_rank:
                continue

            if rank == src_rank:
                p2p_op_list.append(dist.P2POp(dist.isend, send_tensor, recv_rank))

            if rank == recv_rank:
                p2p_op_list.append(dist.P2POp(dist.irecv, recv_tensor, src_rank))
    reqs = dist.batch_isend_irecv(p2p_op_list)

    for req in reqs:
        req.wait()


def _optimized_nccl_scatter_op(
    send_buffer_list: List[Tensor],
    recv_buffer_list: List[Tensor],
    indices: Tensor,
    recv_comm_matrix: Tensor,
    rank: int,
    world_size: int,
) -> List[Tensor]:
    """
    An optimized scatter operation.
    """
    p2p_op_list = []

    for send_rank_index in range(world_size):
        for recv_rank_index in range(world_size):
            if send_rank_index == recv_rank_index:
                # No self-sends allowed. Should be done in the local scatter.
                continue
            if (send_rank_index != rank) or (recv_rank_index != rank):
                # Current rank not involved in this p2p communication pair.
                continue

            if recv_comm_matrix[send_rank_index, recv_rank_index] == 0:
                # No communication between these ranks.
                continue

    reqs = dist.batch_isend_irecv(p2p_op_list)

    for req in reqs:
        req.wait()

    return recv_buffer_list
