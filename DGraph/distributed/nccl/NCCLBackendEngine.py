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
from DGraph.distributed.nccl.scatter_op_impl import _nccl_scatter_op
from DGraph.distributed.nccl.alltoallv_impl import _nccl_alltoall_v
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

        num_local_output_rows = largest_split(indices.shape[1], world_size)
        batch_size = 1
        num_features = local_send_tensor.shape[2]

        recv_tensor = torch.zeros(batch_size, num_local_output_rows, num_features).to(
            local_send_tensor.device
        )
        _start_index = num_local_output_rows * rank
        _end_index = num_local_output_rows * (rank + 1)
        local_indices_slice = indices[0][_start_index:_end_index]

        local_rank_mapping = send_ranks[_start_index:_end_index]

        local_indices = local_indices_slice % local_send_tensor.shape[1]

        print(rank, local_indices_slice)
        print(rank, local_rank_mapping)
        print(local_indices_slice[local_rank_mapping == rank])
        # assert (
        #     local_indices.shape[0] == 4
        # ), f"Incorrect {send_ranks[local_indices_slice].shape} {local_indices.shape}"
        # do local gather if any slices are local
        if len(local_indices_slice) > 0:
            recv_tensor[:, local_rank_mapping == rank, :] = RankLocalMaskedGather(
                local_send_tensor, local_indices, local_rank_mapping, rank
            )

        recv_tensor = _nccl_alltoall_v(
            local_send_tensor=local_send_tensor,
            local_recv_tensor=recv_tensor,
            indices=indices,
            local_rank_mapping=local_rank_mapping,
            src_ranks=send_ranks,
            dest_ranks=recv_ranks,
            rank=rank,
            world_size=world_size,
        )

        return recv_tensor

    @staticmethod
    def backward(ctx, grad_output):
        indices, rank, world_size = ctx.saved_tensors

        # _nccl_scatter_op(grad_output, send_tensor, indices, rank, world_size)
        return grad_output, None, None, None, None, None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ScatterFunction(Function):
    @staticmethod
    def forward(
        ctx,
        send_tensor: torch.Tensor,
        recv_tensor: torch.Tensor,
        indices: torch.Tensor,
        send_ranks: torch.Tensor,
        recv_ranks: torch.Tensor,
        output_size: int,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(
            torch.tensor(output_size),
            torch.tensor(rank),
            torch.tensor(world_size),
        )
        feature_size = send_tensor.shape[1]
        recv_tensor = torch.zeros(output_size, feature_size).to(send_tensor.device)

        # Start local only scatter
        local_indices_slice = indices[send_ranks == rank]
        local_rank_mapping = recv_ranks[local_indices_slice]
        local_indices = local_indices_slice[local_rank_mapping == rank]
        local_send_tensor = send_tensor[local_rank_mapping == rank]
        recv_tensor.scatter_add_(0, local_indices, local_send_tensor)
        # End local only scatter

        _all_messages_mask = send_ranks != recv_ranks
        _remote_sender_ranks = send_ranks[_all_messages_mask]
        _remote_receiver_ranks = recv_ranks[_all_messages_mask]
        _indices = indices[_all_messages_mask]
        # This is not the message size because we have to locally aggregate
        # the messages before sending them to the remote rank

        # Perform local aggregation, also coung the number of messages
        comm_matrix = torch.zeros(world_size, world_size).long()
        recv_buffer_list = []
        send_buffer_list = []
        recv_local_placement = []

        for _sender in range(world_size):
            _sender_mask = _remote_sender_ranks == _sender
            for _receiver in range(world_size):
                if _sender == _receiver:
                    continue
                _receiver_mask = _remote_receiver_ranks == _receiver
                _mask = _sender_mask & _receiver_mask

                if torch.sum(_mask) == 0:
                    # No messages to send
                    continue
                recv_positions = _indices[_mask]
                unique_indices = torch.unique(recv_positions)
                num_messages = len(unique_indices)

                renumbered_indices = torch.zeros_like(_indices)
                # TODO: Optimize this code
                for i, idx in enumerate(unique_indices):
                    renumbered_indices[_indices == idx] = i
                comm_matrix[_sender, _receiver] = num_messages
                recv_buffer = torch.zeros(num_messages, feature_size).to(
                    send_tensor.device
                )
                send_buffer = torch.zeros(num_messages, feature_size).to(
                    send_tensor.device
                )
                send_buffer.scatter_add_(0, renumbered_indices, send_tensor[_mask])
                recv_buffer_list.append(recv_buffer)
                recv_local_placement.append(recv_positions)

        # Communication happens when send_ranks != recv_ranks
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
    _is_initialized = False
    _rank = -1
    _world_size = -1

    def __init__(self, *args, **kwargs):
        # check if already initialized
        # self._initialized = dist.is_initialized()
        if not NCCLBackendEngine._is_initialized:
            self.init_process_group()

    def init_process_group(self, *args, **kwargs):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", *args, **kwargs)

        NCCLBackendEngine._is_initialized = True

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

    def gather(
        self, local_send_tensor, indices, rank_mappings, **kwargs
    ) -> torch.Tensor:

        send_tensor_shape = local_send_tensor.shape
        b_size = send_tensor_shape[0]
        assert b_size == 1, "Multi-batch gather disabled for testing"
        assert len(send_tensor_shape) == 3, "Currently only support 3D tensors"
        assert len(rank_mappings.shape) == 2
        assert rank_mappings.shape[0] == 2
        assert indices.shape[-1] == rank_mappings.shape[-1]

        world_size = self.get_world_size()
        rank = self.get_rank()
        send_rank = rank_mappings[0]
        recv_rank = rank_mappings[1]

        output_tensor = GatherFunction.apply(
            local_send_tensor,
            indices,
            send_rank,
            recv_rank,
            rank,
            world_size,
        )

        return output_tensor  # type: ignore

    def destroy(self) -> None:
        if self._initialized:
            # dist.destroy_process_group()
            self._initialized = False

    def finalize(self) -> None:
        if self._initialized:
            dist.barrier()
