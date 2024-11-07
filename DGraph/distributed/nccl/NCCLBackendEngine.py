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
import torch
import torch.distributed as dist
from DGraph.distributed.Engine import BackendEngine
from DGraph.distributed.nccl.gather_op_impl import (
    _nccl_gather_op,
    _optimized_nccl_gather_op,
)
from DGraph.distributed.nccl.scatter_op_impl import _nccl_scatter_op
from DGraph.distributed.RankLocalOps import RankLocalMaskedGather
from torch.autograd import Function
from DGraph.utils import largest_split


class GatherFunction(Function):
    @staticmethod
    def forward(
        ctx,
        local_send_tensor: torch.Tensor,
        indices: torch.LongTensor,
        send_ranks: torch.Tensor,
        recv_ranks: torch.Tensor,
        rank: int,
        world_size: int,
    ):
        ctx.save_for_backward(
            indices,
            torch.tensor(rank),
            torch.tensor(world_size),
        )

        # Since NCCL is two-sided, we need to push from local rank and pull from
        # remote rank to get the global gather

        # TODO: One possible optmization is cache all these calculations
        # and only do the gather when the cache is invalidated. Essentially
        # if we are working with static graphs, the indices and distribution pattern
        # will not change and we can cache the communication pattern. - S.Z

        # We can also pre-compute this on the data ingestion side. Might
        # be worth looking to some kind of cached communication pattern store
        # that can be passed to the communicator. - S.Z

        recv_tensor = torch.zeros_like(local_send_tensor)
        num_features = local_send_tensor.shape[1]

        local_indices_slice = indices[recv_ranks == rank]
        local_rank_mapping = send_ranks[local_indices_slice]

        local_indices = local_indices_slice[local_rank_mapping == rank]

        # do local gather
        recv_tensor[local_rank_mapping == rank] = RankLocalMaskedGather(
            local_send_tensor, local_indices, local_rank_mapping, rank
        )

        # Communication happens when send_ranks != recv_ranks
        remote_indices_mask = send_ranks != recv_ranks
        remote_sender_ranks = send_ranks[remote_indices_mask]
        remote_receiver_ranks = recv_ranks[remote_indices_mask]

        # Get the messages being sent to this rank

        remote_receiver_rank_here = remote_receiver_ranks == rank
        remote_sender_ranks_here = remote_sender_ranks[remote_receiver_rank_here]

        # For now assume only 1 partition per world size

        comm_vector = torch.bincount(
            remote_sender_ranks_here, minlength=world_size
        ).long()

        recv_buffer_list = []
        recv_local_placement = []

        for i, num_messages in enumerate(comm_vector):
            if num_messages == 0:
                continue
            if i == rank:
                continue

            recv_buffer = torch.zeros(int(num_messages.item()), num_features).to(
                local_send_tensor.device
            )
            recv_buffer_list.append(recv_buffer)
            _local_placement_indices = torch.where(local_rank_mapping == i)[0]
            recv_local_placement.append(_local_placement_indices)

        # do global gather
        remote_messages = _optimized_nccl_gather_op(
            local_send_tensor,
            recv_buffer_list,
            remote_sender_ranks,
            remote_receiver_ranks,
            comm_vector,
            rank,
            world_size,
        )
        for i, recv_buffer in enumerate(remote_messages):
            if comm_vector[i] == 0:
                continue
            recv_tensor[recv_local_placement[i]] = recv_buffer

        return recv_tensor

    @staticmethod
    def backward(ctx, grad_output):
        indices, rank, world_size = ctx.saved_tensors

        # _nccl_scatter_op(grad_output, send_tensor, indices, rank, world_size)
        return grad_output, None, None


class ScatterFunction(Function):
    @staticmethod
    def forward(
        ctx,
        send_tensor: torch.Tensor,
        recv_tensor: torch.Tensor,
        indices: torch.Tensor,
        global_rank_mapping: torch.Tensor,
        local_indices: torch.Tensor,
        local_rank_mapping: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(
            send_tensor,
            recv_tensor,
            indices,
            torch.tensor(rank),
            torch.tensor(world_size),
        )
        recv_tensor.scatter_add_(0, local_indices, send_tensor)
        _nccl_scatter_op(send_tensor, recv_tensor, indices, rank, world_size)
        return recv_tensor

    @staticmethod
    def backward(ctx, grad_output):
        send_tensor, recv_tensor, indices = ctx.saved_tensors
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        _nccl_scatter_op(grad_output, send_tensor, indices, rank, world_size)
        return grad_output, None, None


def scatter(
    send_tensor: torch.Tensor, recv_tensor: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    return ScatterFunction.apply(send_tensor, recv_tensor, indices)  # type: ignore


def gather(send_tensor, recv_tensor, indices) -> torch.Tensor:
    return GatherFunction.apply(send_tensor, recv_tensor, indices)  # type: ignore


class NCCLBackendEngine(BackendEngine):
    def __init__(self, *args, **kwargs):
        # check if already initialized
        self._initialized = dist.is_initialized()

    def init_process_group(self, *args, **kwargs):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", *args, **kwargs)

        self._initialized = True

    def get_rank(self) -> int:
        return dist.get_rank()

    def get_world_size(self) -> int:
        return dist.get_world_size()

    def get_local_rank_slice(self, tensor: torch.Tensor) -> torch.Tensor:
        rank = self.get_rank()
        world_size = self.get_world_size()
        tensor_shape = tensor.shape
        tensor_size = tensor_shape[1]
        local_size = tensor_size // world_size
        start_index = rank * local_size
        end_index = start_index + local_size
        return tensor[:, start_index:end_index]

    def scatter(
        self, src_tensor, indices, rank_mappings, output_size, *args, **kwargs
    ) -> torch.Tensor:
        input_tensor: torch.Tensor = args[0]
        batch_size: int = src_tensor.shape[0]
        feature_size: int = src_tensor.shape[2]

        output_tensor: torch.Tensor = torch.zeros(
            (
                batch_size,
                output_size,
                feature_size,
            )
        )
        return scatter(src_tensor, output_tensor, indices)

    def gather(self, *args, **kwargs) -> torch.Tensor:
        input_tensor = args[0]
        indices = args[1]
        indices_shape = indices.shape
        b_size = indices_shape[0]

        assert b_size == 1, "Multi-batch gather disabled for testing"
        world_size = self.get_world_size()
        rank = self.get_rank()
        num_local_output_rows = largest_split(indices_shape[1], world_size)
        num_local_input_rows = largest_split(input_tensor.shape[1], world_size)
        send_rank = args[2]
        recv_rank = torch.arange(len(indices)) // num_local_output_rows

        renumbered_indices = indices % num_local_input_rows
        output_tensor = GatherFunction.apply(
            input_tensor, renumbered_indices, send_rank, recv_rank, rank, world_size
        )

        return output_tensor  # type: ignore

    def destroy(self) -> None:
        if self._initialized:
            # dist.destroy_process_group()
            self._initialized = False

    def finalize(self) -> None:
        if self._initialized:
            dist.barrier()
